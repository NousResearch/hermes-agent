import { useStore } from '@nanostores/react'
import * as React from 'react'

import {
  getCronJobs,
  getSkills,
  listAllProfileSessions,
  listMcpServers,
  pauseCronJob,
  resumeCronJob,
  setMcpServerEnabled,
  toggleSkill
} from '@/hermes'
import { broadcastDesktopStateChange } from '@/lib/desktop-state-sync'
import {
  COMMUNITY_LEARN_LINKS,
  isAllowlistedLearnUrl,
  type LearnLink,
  MENU_BAR_SHOW_COMMUNITY_LEARN_LINKS,
  OFFICIAL_LEARN_LINKS
} from '@/lib/menu-bar-learn-links'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import type { CronJob, McpServerSummary, SessionInfo, SkillInfo } from '@/types/hermes'

async function openLearnLink(link: LearnLink) {
  if (!isAllowlistedLearnUrl(link.url)) {
    throw new Error(`Refusing non-allowlisted learn URL: ${link.url}`)
  }

  await window.hermesDesktop.openExternal(link.url)
}

function LearnCard({ link, onOpen }: { link: LearnLink; onOpen: (link: LearnLink) => void }) {
  return (
    <button className="mbc-learn-card" data-kind={link.kind} onClick={() => onOpen(link)} type="button">
      <span className="mbc-learn-title">{link.title}</span>
      <span className="mbc-learn-desc">{link.description}</span>
      {link.kind === 'community' ? (
        <span className="mbc-learn-badge">Community · unofficial · not affiliated with Nous Research</span>
      ) : (
        <span className="mbc-learn-badge mbc-learn-badge-official">Official</span>
      )}
    </button>
  )
}

export function AtlasTab() {
  const activeProfile = normalizeProfileKey(useStore($activeGatewayProfile))
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState('')
  const [busy, setBusy] = React.useState(false)
  const [sessions, setSessions] = React.useState<SessionInfo[]>([])
  const [skills, setSkills] = React.useState<SkillInfo[]>([])
  const [mcp, setMcp] = React.useState<McpServerSummary[]>([])
  const [cron, setCron] = React.useState<CronJob[]>([])
  const [status, setStatus] = React.useState('')
  const [skillFilter, setSkillFilter] = React.useState('')

  const refresh = React.useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const [sessionPage, skillList, mcpList, cronJobs] = await Promise.all([
        listAllProfileSessions(40, 0, 'exclude', 'recent', activeProfile),
        getSkills(),
        listMcpServers().catch(() => ({ servers: [] as McpServerSummary[] })),
        getCronJobs().catch(() => [] as CronJob[])
      ])

      setSessions(sessionPage.sessions || [])
      // Full installed skill set — no artificial slice; UI scrolls
      setSkills((skillList || []).slice().sort((a, b) => a.name.localeCompare(b.name)))
      setMcp(mcpList.servers || [])
      setCron(cronJobs || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [activeProfile])

  React.useEffect(() => {
    void refresh()
  }, [refresh])

  const filteredSkills = React.useMemo(() => {
    const needle = skillFilter.trim().toLowerCase()

    if (!needle) {
      return skills
    }

    return skills.filter(
      skill =>
        skill.name.toLowerCase().includes(needle) ||
        skill.description?.toLowerCase().includes(needle) ||
        skill.category?.toLowerCase().includes(needle)
    )
  }, [skills, skillFilter])

  const openSession = async (sessionId: string) => {
    setStatus('')

    try {
      const result = await window.hermesDesktop.openSessionWindow(sessionId)

      if (!result.ok) {
        throw new Error(result.error || 'Could not open session')
      }

      setStatus(`Opened session ${sessionId.slice(0, 8)}…`)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
    }
  }

  const onToggleSkill = async (skill: SkillInfo) => {
    setBusy(true)
    setStatus('')

    try {
      const next = !skill.enabled
      const result = await toggleSkill(skill.name, next)
      setSkills(prev => prev.map(s => (s.name === skill.name ? { ...s, enabled: result.enabled } : s)))
      broadcastDesktopStateChange('skills', {
        profile: activeProfile,
        value: { enabled: result.enabled, name: skill.name }
      })
      // proof
      const proof = await getSkills()
      const found = proof.find(s => s.name === skill.name)

      if (found && found.enabled !== result.enabled) {
        throw new Error('Skill toggle did not stick')
      }

      setStatus(`${skill.name} ${result.enabled ? 'enabled' : 'disabled'}`)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
      await refresh()
      broadcastDesktopStateChange('skills', { profile: activeProfile })
    } finally {
      setBusy(false)
    }
  }

  const onToggleMcp = async (server: McpServerSummary) => {
    setBusy(true)
    setStatus('')

    try {
      const next = !server.enabled
      await setMcpServerEnabled(server.name, next)
      setMcp(prev => prev.map(row => (row.name === server.name ? { ...row, enabled: next } : row)))
      broadcastDesktopStateChange('mcp', {
        profile: activeProfile,
        value: { enabled: next, name: server.name }
      })
      const proof = await listMcpServers()
      const found = proof.servers.find(s => s.name === server.name)

      if (!found || found.enabled !== next) {
        throw new Error('MCP toggle did not stick')
      }

      setMcp(proof.servers)
      setStatus(`${server.name} ${next ? 'enabled' : 'disabled'}`)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
      await refresh()
      broadcastDesktopStateChange('mcp', { profile: activeProfile })
    } finally {
      setBusy(false)
    }
  }

  const onToggleCron = async (job: CronJob) => {
    setBusy(true)
    setStatus('')

    try {
      const expectedEnabled = !job.enabled
      const changed = await (job.enabled ? pauseCronJob(job.id) : resumeCronJob(job.id))
      setCron(prev => prev.map(candidate => (candidate.id === job.id ? changed : candidate)))
      broadcastDesktopStateChange('cron', { profile: activeProfile, value: changed })
      const proof = await getCronJobs()
      const updated = proof.find(candidate => candidate.id === job.id)

      if (!updated || Boolean(updated.enabled) !== expectedEnabled) {
        throw new Error('Cron state change did not stick')
      }

      setCron(proof)
      setStatus(`${updated.name || updated.id} ${updated.enabled ? 'resumed' : 'paused'}`)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
      await refresh()
      broadcastDesktopStateChange('cron', { profile: activeProfile })
    } finally {
      setBusy(false)
    }
  }

  const onOpenLearnLink = (link: LearnLink) => {
    setStatus('')
    void openLearnLink(link)
      .then(() => setStatus(`Opened ${link.title}`))
      .catch(err => setStatus(err instanceof Error ? err.message : String(err)))
  }

  return (
    <div className="mbc-tab-panel mbc-stack" data-tab="atlas">
      <section className="mbc-card">
        <div className="mbc-card-head">
          <h3>Local map</h3>
          <button className="mbc-button" disabled={loading || busy} onClick={() => void refresh()} type="button">
            Refresh
          </button>
        </div>
        {loading ? <p className="mbc-muted">Loading Desktop state…</p> : null}
        {error ? <p className="mbc-muted">Error: {error}</p> : null}
      </section>

      <section className="mbc-card">
        <h3>Recent sessions ({sessions.length})</h3>
        <div aria-label="Sessions" className="mbc-scroll-list">
          {sessions.length === 0 ? (
            <p className="mbc-muted">No sessions.</p>
          ) : (
            sessions.map(session => (
              <button
                className="mbc-list-button"
                key={session.id}
                onClick={() => void openSession(session.id)}
                type="button"
              >
                {session.title || session.id.slice(0, 12)}
                {session.is_active ? ' · active' : ''}
              </button>
            ))
          )}
        </div>
      </section>

      <section className="mbc-card">
        <h3>Skills ({skills.length})</h3>
        <input
          aria-label="Filter skills"
          className="mbc-search"
          onChange={event => setSkillFilter(event.target.value)}
          placeholder="Filter skills…"
          type="search"
          value={skillFilter}
        />
        <div aria-label="Installed skills" className="mbc-scroll-list mbc-scroll-list-tall">
          {filteredSkills.length === 0 ? (
            <p className="mbc-muted">No skills match.</p>
          ) : (
            filteredSkills.map(skill => (
              <label className="mbc-toggle-row" key={skill.name}>
                <input
                  checked={skill.enabled}
                  disabled={busy}
                  onChange={() => void onToggleSkill(skill)}
                  type="checkbox"
                />
                <span className="mbc-toggle-copy">
                  <strong>{skill.name}</strong>
                  <em>{skill.category || 'uncategorized'}</em>
                </span>
              </label>
            ))
          )}
        </div>
      </section>

      <section className="mbc-card">
        <h3>MCP servers ({mcp.length})</h3>
        <div aria-label="MCP servers" className="mbc-scroll-list">
          {mcp.length === 0 ? (
            <p className="mbc-muted">No MCP servers configured.</p>
          ) : (
            mcp.map(server => (
              <label className="mbc-toggle-row" key={server.name}>
                <input
                  checked={server.enabled}
                  disabled={busy}
                  onChange={() => void onToggleMcp(server)}
                  type="checkbox"
                />
                <span className="mbc-toggle-copy">
                  <strong>{server.name}</strong>
                  <em>{server.transport}</em>
                </span>
              </label>
            ))
          )}
        </div>
      </section>

      <section className="mbc-card">
        <h3>Cron jobs ({cron.length})</h3>
        <div aria-label="Cron jobs" className="mbc-scroll-list">
          {cron.length === 0 ? (
            <p className="mbc-muted">No cron jobs.</p>
          ) : (
            cron.map(job => (
              <label className="mbc-toggle-row" key={job.id}>
                <input checked={job.enabled} disabled={busy} onChange={() => void onToggleCron(job)} type="checkbox" />
                <span className="mbc-toggle-copy">
                  <strong>{job.name || job.id}</strong>
                  <em>{job.schedule_display || job.state || (job.enabled ? 'enabled' : 'paused')}</em>
                </span>
              </label>
            ))
          )}
        </div>
      </section>

      <section aria-labelledby="mbc-learn-heading" className="mbc-card">
        <h3 id="mbc-learn-heading">Learn & Ecosystem</h3>
        <p className="mbc-muted">Official resources first. Community sites open externally only.</p>
        <div aria-label="Official resources" className="mbc-learn-grid">
          {OFFICIAL_LEARN_LINKS.map(link => (
            <LearnCard key={link.id} link={link} onOpen={onOpenLearnLink} />
          ))}
        </div>
        {MENU_BAR_SHOW_COMMUNITY_LEARN_LINKS ? (
          <>
            <h4 className="mbc-subhead">Community</h4>
            <p className="mbc-muted mbc-tight">Unofficial · not Nous documentation.</p>
            <div aria-label="Community resources" className="mbc-learn-grid">
              {COMMUNITY_LEARN_LINKS.map(link => (
                <LearnCard key={link.id} link={link} onOpen={onOpenLearnLink} />
              ))}
            </div>
          </>
        ) : null}
      </section>

      {status ? <p className="mbc-status">{status}</p> : null}
    </div>
  )
}
