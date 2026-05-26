(function () {
  const sdk = window.__HERMES_PLUGIN_SDK__;
  const plugins = window.__HERMES_PLUGINS__;
  if (!sdk || !plugins) return;
  const React = sdk.React;
  const h = React.createElement;
  const fetchJSON = sdk.fetchJSON;
  const Card = sdk.components.Card;
  const CardHeader = sdk.components.CardHeader;
  const CardTitle = sdk.components.CardTitle;
  const CardContent = sdk.components.CardContent;
  const Badge = sdk.components.Badge;

  function ActionButton(props) {
    const disabled = props && props.disabled;
    const extraClass = props && props.className ? ' ' + props.className : '';
    const className = 'visibility-os-action-button inline-flex items-center justify-center rounded border border-current/20 bg-midground px-2 py-1 text-xs font-semibold leading-none tracking-normal normal-case whitespace-nowrap hover:bg-midground/90 disabled:opacity-50 disabled:cursor-not-allowed' + extraClass;
    const style = Object.assign({ color: '#061512', letterSpacing: 'normal', textTransform: 'none', fontFamily: 'Arial, Helvetica, sans-serif', fontWeight: 700, lineHeight: 1.1 }, props && props.style ? props.style : {});
    return h('button', Object.assign({}, props || {}, { type: (props && props.type) || 'button', disabled: disabled, className: className, style: style }), props && props.children);
  }

  function payloadText(action) {
    const p = action.final_payload || action.proposed_payload || {};
    return p.text || p.body || JSON.stringify(p, null, 2);
  }

  function VisibilityOS() {
    const hooks = sdk.hooks;
    const [feed, setFeed] = hooks.useState({ items: [], counts: {} });
    const [error, setError] = hooks.useState(null);
    const [busy, setBusy] = hooks.useState(false);
    const [selectedOpportunity, setSelectedOpportunity] = hooks.useState(null);
    const [selectedWorkstream, setSelectedWorkstream] = hooks.useState(null);
    const [selectedTicket, setSelectedTicket] = hooks.useState(null);
    const [slackTarget, setSlackTarget] = hooks.useState('');
    const [severityFilter, setSeverityFilter] = hooks.useState('all');
    const [sourceFilter, setSourceFilter] = hooks.useState('all');
    const [showArchived, setShowArchived] = hooks.useState(false);
    const [findingStatus, setFindingStatus] = hooks.useState({});
    const load = hooks.useCallback(function () {
      setBusy(true);
      fetchJSON('/api/plugins/visibility-os/feed' + (showArchived ? '?include_archived=true' : ''))
        .then(setFeed)
        .catch(function (e) { setError(String(e)); })
        .finally(function () { setBusy(false); });
    }, [showArchived]);
    hooks.useEffect(load, [load]);
    hooks.useEffect(function () {
      fetchJSON('/api/plugins/visibility-os/config')
        .then(function (cfg) { if (cfg && cfg.default_slack_channel) setSlackTarget(cfg.default_slack_channel); })
        .catch(function () {});
    }, []);
    function post(path, body) {
      setBusy(true);
      fetchJSON(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })
        .then(load)
        .catch(function (e) { setError(String(e)); setBusy(false); });
    }
    function moveBoardState(item, boardState) {
      post('/api/plugins/visibility-os/board-state', { item_kind: item.kind, item_id: item.id, board_state: boardState, actor: 'human' });
    }
    function viewOpportunity(id) {
      setBusy(true);
      fetchJSON('/api/plugins/visibility-os/opportunities/' + id)
        .then(function (detail) { setSelectedOpportunity(detail); })
        .catch(function (e) { setError(String(e)); })
        .finally(function () { setBusy(false); });
    }
    function viewWorkstream(id) {
      if (!id) return;
      setBusy(true);
      fetchJSON('/api/plugins/visibility-os/workstreams/' + id)
        .then(function (detail) { setSelectedWorkstream(detail); })
        .catch(function (e) { setError(String(e)); })
        .finally(function () { setBusy(false); });
    }
    function draftOpportunity(detail, actionKind) {
      const body = { action_kind: actionKind, actor: 'human' };
      if (actionKind === 'slack_update' && slackTarget) body.target_location = slackTarget;
      post('/api/plugins/visibility-os/opportunities/' + detail.id + '/draft-action', body);
      setSelectedOpportunity(null);
    }
    function auditOpportunity(detail) {
      post('/api/plugins/visibility-os/opportunities/' + detail.id + '/audit-pr', { actor: 'human' });
      setSelectedOpportunity(null);
    }
    function deepReviewOpportunity(detail) {
      post('/api/plugins/visibility-os/opportunities/' + detail.id + '/deep-review-pr', { actor: 'human' });
      setSelectedOpportunity(null);
    }
    function fixCI(item) {
      const opportunityId = item.opportunity_id || item.id;
      post('/api/plugins/visibility-os/opportunities/' + opportunityId + '/fix-ci', { actor: 'human' });
    }
    function fixIssue(item) {
      const opportunityId = item.opportunity_id || item.id;
      post('/api/plugins/visibility-os/opportunities/' + opportunityId + '/fix-issue', { actor: 'human' });
    }
    function pushBranchNow(item) {
      post('/api/plugins/visibility-os/actions/' + item.id + '/approve', { actor: 'human', execute_immediately: true });
    }
    function scoreLine(detail) {
      return 'Impact ' + detail.impact_score + ' · Visibility ' + detail.visibility_score + ' · Effort ' + detail.effort_score + ' · Safety ' + detail.safety_score + ' · Risk penalty ' + detail.risk_penalty;
    }
    function editPayload(item) {
      const current = payloadText(item);
      const next = window.prompt('Edit reviewed payload before approval', current);
      if (next === null) return;
      const key = (item.proposed_payload && item.proposed_payload.body !== undefined) ? 'body' : 'text';
      post('/api/plugins/visibility-os/actions/' + item.id + '/edit', { actor: 'human', final_payload: { [key]: next } });
    }
    function evidenceLinks(item) {
      const links = item.evidence_links || [];
      if (!links.length) return h('div', { className: 'text-xs text-amber-300' }, 'No evidence links attached');
      return h('div', { className: 'flex gap-2 flex-wrap' }, links.map(function (link, idx) {
        const url = link.url || link.href || String(link);
        return h('a', { key: idx, className: 'text-xs underline text-midground', href: url, target: '_blank' }, link.type ? link.type + ': ' + url : url);
      }));
    }
    function displayValue(value) {
      if (value === null || value === undefined) return '';
      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return String(value);
      if (Array.isArray(value)) return value.map(displayValue).filter(Boolean).join(', ');
      if (typeof value === 'object') {
        if (value.path && value.change) return value.path + ' — ' + value.change;
        if (value.path && value.summary) return value.path + ' — ' + value.summary;
        if (value.path) return String(value.path);
        if (value.file) return String(value.file);
        if (value.name) return String(value.name);
        return JSON.stringify(value);
      }
      return String(value);
    }
    function listItems(values) {
      return (values || []).map(function (value, idx) {
        const text = displayValue(value);
        return h('li', { key: text + ':' + idx }, text);
      });
    }
    function proposedPRView(item) {
      if (item.action_type !== 'github_push_branch') return null;
      const p = item.final_payload || item.proposed_payload || {};
      return h('div', { className: 'rounded border border-emerald-500/30 bg-emerald-500/10 p-4 space-y-3' },
        h('div', { className: 'flex items-center justify-between gap-2 flex-wrap' },
          h('div', null,
            h('div', { className: 'text-sm font-bold text-midground' }, 'Proposed PR'),
            h('div', { className: 'text-xs text-text-secondary' }, 'Hermes prepared this locally. Nothing is pushed until you choose Push branch.')
          ),
          h(Badge, null, p.branch || 'branch pending')
        ),
        h('div', { className: 'grid grid-cols-1 md:grid-cols-2 gap-3 text-xs' },
          h('div', { className: 'rounded bg-black/20 p-2' }, h('div', { className: 'font-semibold text-text-secondary' }, 'Commit message'), h('div', null, p.commit_message || 'not provided')),
          h('div', { className: 'rounded bg-black/20 p-2' }, h('div', { className: 'font-semibold text-text-secondary' }, 'PR title'), h('div', null, p.pr_title || p.commit_message || 'not provided'))
        ),
        p.changed_files && p.changed_files.length && h('div', { className: 'text-xs' }, h('div', { className: 'font-semibold text-text-secondary' }, 'Changed files'), h('ul', { className: 'list-disc pl-4' }, listItems(p.changed_files))),
        p.verification && p.verification.length && h('div', { className: 'text-xs' }, h('div', { className: 'font-semibold text-text-secondary' }, 'Verification'), h('ul', { className: 'list-disc pl-4' }, listItems(p.verification))),
        p.self_audit && h('div', { className: 'text-xs rounded bg-black/20 p-2' }, h('div', { className: 'font-semibold text-text-secondary' }, 'Self-audit'), h('div', null, 'Status: ' + (p.self_audit.audit_status || 'unknown')), p.self_audit.issues_found && h('div', null, 'Issues found: ' + p.self_audit.issues_found.length), p.self_audit.fixes_applied && h('div', null, 'Fixes applied: ' + p.self_audit.fixes_applied.length)),
        p.independent_review && h('div', { className: 'text-xs rounded bg-black/20 p-2' }, h('div', { className: 'font-semibold text-text-secondary' }, 'Independent review'), h('div', null, 'Status: ' + (p.independent_review.review_status || 'unknown')), p.independent_review.findings && h('div', null, 'Findings: ' + p.independent_review.findings.length), p.independent_review.fixes_required && h('div', null, 'Fixes required: ' + p.independent_review.fixes_required.length)),
        h('details', { className: 'text-xs' }, h('summary', { className: 'cursor-pointer font-semibold text-text-secondary' }, 'PR body'), h('pre', { className: 'mt-2 whitespace-pre-wrap rounded bg-black/40 p-3' }, p.pr_body || 'not provided'))
      );
    }
    function actionPayloadView(item) {
      if (item.action_type === 'github_push_branch') return proposedPRView(item);
      return h('details', { className: 'text-xs' },
        h('summary', { className: 'cursor-pointer text-text-secondary' }, 'Raw payload'),
        h('pre', { className: 'mt-2 whitespace-pre-wrap rounded bg-black/40 p-3 text-xs' }, payloadText(item))
      );
    }
    function progressBar(value) {
      const pct = Math.max(0, Math.min(100, Number(value || 0)));
      return h('div', { className: 'h-2 rounded bg-black/30 overflow-hidden' }, h('div', { className: 'h-full bg-emerald-400', style: { width: pct + '%' } }));
    }
    function workstreamBadge(item) {
      if (!item.workstream_id) return h(Badge, null, 'Not started');
      return h(ActionButton, { onClick: function () { viewWorkstream(item.workstream_id); }, className: 'border-emerald-500/40' }, (item.workstream_stage || 'workstream').replace(/_/g, ' '));
    }
    function workstreamDetailView(ws) {
      if (!ws) return null;
      const artifacts = ws.artifacts || [];
      const proposed = artifacts.find(function (a) { return a.artifact_type === 'proposed_pr'; });
      const diff = artifacts.find(function (a) { return a.artifact_type === 'diff_summary'; });
      const review = artifacts.find(function (a) { return a.artifact_type === 'review_findings'; });
      return h(Card, { className: 'border-emerald-500/40' },
        h(CardHeader, null, h(CardTitle, null, 'Workstream: ' + ws.title)),
        h(CardContent, { className: 'space-y-4' },
          h('div', { className: 'flex gap-2 flex-wrap' }, h(Badge, null, ws.status), h(Badge, null, (ws.stage || '').replace(/_/g, ' ')), ws.repo && h(Badge, null, ws.repo), h(Badge, null, ws.lane_kind)),
          h('div', { className: 'space-y-1' }, h('div', { className: 'text-xs text-text-secondary' }, ws.current_step || 'No current step recorded'), progressBar(ws.progress_percent)),
          ws.source_url && h('a', { className: 'text-xs underline text-midground', href: ws.source_url, target: '_blank' }, ws.source_url),
          h('div', { className: 'grid grid-cols-1 md:grid-cols-2 gap-3' },
            h('div', { className: 'rounded border border-current/10 p-3 space-y-2' },
              h('div', { className: 'text-sm font-bold text-midground' }, 'Timeline'),
              (ws.events || []).map(function (ev) { return h('div', { key: ev.id, className: 'text-xs rounded bg-black/20 p-2' }, h('div', { className: 'font-semibold' }, (ev.stage || ev.event_type || '').replace(/_/g, ' ')), h('div', { className: 'text-text-secondary' }, ev.message), h('div', { className: 'text-[10px] text-text-secondary' }, ev.created_at)); })
            ),
            h('div', { className: 'rounded border border-current/10 p-3 space-y-2' },
              h('div', { className: 'text-sm font-bold text-midground' }, 'Artifacts'),
              !artifacts.length && h('div', { className: 'text-xs text-text-secondary' }, 'No artifacts yet'),
              artifacts.map(function (a) { return h('details', { key: a.id, className: 'text-xs rounded bg-black/20 p-2' }, h('summary', { className: 'cursor-pointer font-semibold' }, a.artifact_type + ': ' + a.title), h('div', { className: 'text-text-secondary py-1' }, a.summary), h('pre', { className: 'whitespace-pre-wrap rounded bg-black/40 p-2' }, JSON.stringify(a.payload || {}, null, 2))); })
            )
          ),
          proposed && h('div', { className: 'rounded border border-emerald-500/30 bg-emerald-500/10 p-3 space-y-2 text-xs' },
            h('div', { className: 'text-sm font-bold text-midground' }, 'Proposed PR'),
            h('div', null, 'Branch: ' + ((proposed.payload || {}).branch || 'unknown')),
            h('div', null, 'Title: ' + ((proposed.payload || {}).pr_title || 'not provided')),
            (proposed.payload || {}).changed_files && h('ul', { className: 'list-disc pl-4' }, listItems(proposed.payload.changed_files || []))
          ),
          diff && h('div', { className: 'rounded border border-current/10 p-3 text-xs' }, h('div', { className: 'text-sm font-bold text-midground' }, 'Change summary'), h('ul', { className: 'list-disc pl-4' }, listItems((diff.payload || {}).changed_files || []))),
          review && h('div', { className: 'rounded border border-current/10 p-3 text-xs' }, h('div', { className: 'text-sm font-bold text-midground' }, 'Review findings'), h('pre', { className: 'whitespace-pre-wrap rounded bg-black/40 p-2' }, JSON.stringify(review.payload || {}, null, 2))),
          h(ActionButton, { onClick: function () { setSelectedWorkstream(null); } }, 'Close workstream')
        )
      );
    }
    function modalShell(title, subtitle, onClose, content) {
      return h('div', { className: 'visibility-os-modal-backdrop', role: 'presentation', onMouseDown: function (e) { if (e.target === e.currentTarget) onClose(); } },
        h('div', { className: 'visibility-os-modal', role: 'dialog', 'aria-modal': 'true', 'aria-label': title },
          h('div', { className: 'visibility-os-modal-header' },
            h('div', null,
              h('div', { className: 'visibility-os-modal-kicker' }, subtitle || 'Visibility OS ticket'),
              h('h2', { className: 'visibility-os-modal-title' }, title)
            ),
            h(ActionButton, { onClick: onClose, className: 'visibility-os-modal-close' }, 'Close')
          ),
          h('div', { className: 'visibility-os-modal-body' }, content)
        )
      );
    }
    function ticketModalView(item) {
      if (!item) return null;
      const title = item.title || item.opportunity_title || item.summary || item.id;
      const description = item.kind === 'opportunity' ? item.description : item.summary;
      return modalShell(title, item.kind === 'opportunity' ? 'Opportunity' : 'Action', function () { setSelectedTicket(null); },
        h('div', { className: 'space-y-4' },
          h('div', { className: 'flex gap-2 flex-wrap' },
            h(Badge, null, item.kind || 'ticket'),
            item.repo && h(Badge, null, item.repo),
            item.source_repo && h(Badge, null, item.source_repo),
            item.category && h(Badge, null, item.category),
            item.status && h(Badge, null, item.status),
            item.priority_score && h(Badge, null, 'Priority ' + item.priority_score),
            item.board_state && h(Badge, null, item.board_state.replace(/_/g, ' '))
          ),
          description && h('p', { className: 'text-sm text-text-secondary' }, description),
          item.current_step && h('div', { className: 'rounded border border-current/10 bg-black/20 p-3 text-xs' }, item.current_step),
          item.source_url && h('a', { className: 'text-xs underline text-midground', href: item.source_url, target: '_blank' }, item.source_url),
          item.workstream_id && h('div', { className: 'text-xs' }, workstreamBadge(item)),
          item.kind === 'action' && actionPayloadView(item),
          item.kind === 'action' && evidenceLinks(item),
          item.action_type === 'github_push_branch' && h('div', { className: 'text-xs text-text-secondary' }, 'Review the prepared PR here, then use Push branch when you are ready.'),
          findingsView(item),
          h('div', { className: 'visibility-os-modal-actions' }, sectionActions(item))
        )
      );
    }
    function opportunityModalView(detail) {
      if (!detail) return null;
      return modalShell('Opportunity detail: ' + detail.title, detail.source_repo || detail.source_system || 'Opportunity', function () { setSelectedOpportunity(null); },
        h('div', { className: 'space-y-3' },
          h('div', { className: 'flex gap-2 flex-wrap' },
            h(Badge, null, detail.source_repo || detail.source_system),
            h(Badge, null, detail.category),
            h(Badge, null, 'Priority ' + detail.priority_score)
          ),
          h('p', { className: 'text-sm text-text-secondary' }, detail.why_it_matters || detail.description),
          h('p', { className: 'text-xs text-text-secondary' }, scoreLine(detail)),
          detail.source_url && h('a', { className: 'text-xs underline text-midground', href: detail.source_url, target: '_blank' }, detail.source_url),
          detail.score_explanation && h('pre', { className: 'whitespace-pre-wrap rounded bg-black/40 p-3 text-xs' }, detail.score_explanation),
          evidenceLinks(detail),
          h('div', { className: 'flex items-center gap-2 flex-wrap' },
            h('label', { className: 'text-xs text-text-secondary' }, 'Slack target'),
            h('input', { className: 'rounded bg-black/40 px-2 py-1 text-xs border border-current/20', value: slackTarget, onChange: function (e) { setSlackTarget(e.target.value); } })
          ),
          h('div', { className: 'visibility-os-modal-actions' },
            detail.source_url && detail.source_url.indexOf('/pull/') !== -1 && h(ActionButton, { disabled: busy, onClick: function () { auditOpportunity(detail); } }, 'Audit PR'),
            detail.source_url && detail.source_url.indexOf('/pull/') !== -1 && h(ActionButton, { disabled: busy, onClick: function () { deepReviewOpportunity(detail); } }, 'Deep Review PR'),
            (detail.recommended_actions || []).map(function (a) {
              return h(ActionButton, { key: a.action_kind, disabled: busy, onClick: function () { draftOpportunity(detail, a.action_kind); } }, a.label);
            })
          )
        )
      );
    }
    function findingsSummaryView(findings) {
      const counts = findings.reduce(function (acc, f) { acc[f.severity || 'suggestion'] = (acc[f.severity || 'suggestion'] || 0) + 1; return acc; }, { critical: 0, warning: 0, suggestion: 0 });
      return h('div', { className: 'grid grid-cols-3 gap-2 text-xs' },
        ['critical', 'warning', 'suggestion'].map(function (sev) {
          return h('div', { key: sev, className: 'rounded border border-current/10 bg-black/20 p-2' },
            h('div', { className: 'text-lg font-bold' }, counts[sev] || 0),
            h('div', { className: 'text-text-secondary capitalize' }, sev)
          );
        })
      );
    }
    function groupFindingsByFile(findings) {
      return findings.reduce(function (acc, f) {
        const key = f.path || 'unknown';
        (acc[key] = acc[key] || []).push(f);
        return acc;
      }, {});
    }
    function copyFindingComment(f) {
      const text = f.copy_comment || ((f.casual_comment || f.title || 'Finding') + '\n\n' + (f.path || 'unknown') + ':' + (f.line || '?') + '\n\nSuggested change: ' + (f.solution || ''));
      if (navigator.clipboard) navigator.clipboard.writeText(text);
      else window.prompt('Copy review comment', text);
    }
    function markFindingStatus(item, idx, status) {
      const key = item.id + ':' + idx;
      setFindingStatus(function (prev) { return Object.assign({}, prev, { [key]: status }); });
    }
    function findingCard(item, f, idx) {
      const status = findingStatus[item.id + ':' + idx] || f.status || 'open';
      return h('div', { key: idx, className: 'rounded border border-current/10 bg-black/20 p-3 text-xs space-y-2' },
        h('div', { className: 'flex gap-2 flex-wrap items-center' },
          h(Badge, null, f.severity || 'suggestion'),
          h(Badge, null, f.source || 'deterministic'),
          h(Badge, null, status),
          h(Badge, null, (f.path || 'unknown') + ':' + (f.line || '?'))
        ),
        h('div', { className: 'rounded bg-emerald-500/10 border border-emerald-500/20 p-2 text-sm font-medium' }, f.casual_comment || f.title),
        h('details', { className: 'space-y-2' },
          h('summary', { className: 'cursor-pointer text-text-secondary' }, 'Structured detail'),
          h('div', { className: 'font-semibold pt-2' }, f.title),
          h('div', { className: 'text-text-secondary' }, f.detail),
          h('div', null, h('span', { className: 'text-text-secondary' }, 'Suggested fix: '), f.solution)
        ),
        f.code && h('pre', { className: 'whitespace-pre-wrap rounded bg-black/40 p-2 border-l-2 border-midground' }, f.code),
        h('div', { className: 'flex gap-2 flex-wrap' },
          f.github_diff_url && h('a', { className: 'text-xs underline text-midground', href: f.github_diff_url, target: '_blank' }, 'Open diff line'),
          h(ActionButton, { onClick: function () { copyFindingComment(f); } }, 'Copy comment'),
          h(ActionButton, { onClick: function () { markFindingStatus(item, idx, 'checked'); } }, 'Mark checked'),
          h(ActionButton, { onClick: function () { markFindingStatus(item, idx, 'needs follow-up'); } }, 'Needs follow-up')
        )
      );
    }
    function findingsView(item) {
      const p = item.final_payload || item.proposed_payload || {};
      const findings = p.findings || [];
      if (!findings.length) return null;
      const filtered = findings.filter(function (f) {
        return (severityFilter === 'all' || f.severity === severityFilter) && (sourceFilter === 'all' || (f.source || 'deterministic') === sourceFilter);
      });
      const bySource = {
        agentic: filtered.filter(function (f) { return f.source === 'agentic'; }),
        deterministic: filtered.filter(function (f) { return (f.source || 'deterministic') !== 'agentic'; })
      };
      function sourceSection(title, rows) {
        const grouped = groupFindingsByFile(rows);
        return h('div', { className: 'space-y-2' },
          h('div', { className: 'text-xs font-bold text-midground' }, title + ' (' + rows.length + ')'),
          Object.keys(grouped).sort().map(function (file) {
            return h('details', { key: file, open: true, className: 'rounded border border-current/10 p-2 space-y-2' },
              h('summary', { className: 'cursor-pointer text-xs font-semibold' }, file + ' · ' + grouped[file].length + ' finding(s)'),
              h('div', { className: 'space-y-2 pt-2' }, grouped[file].map(function (f) { return findingCard(item, f, findings.indexOf(f)); }))
            );
          })
        );
      }
      return h('div', { className: 'space-y-3 rounded border border-current/10 p-3' },
        h('div', { className: 'flex items-center justify-between gap-2 flex-wrap' },
          h('div', { className: 'text-sm font-bold text-midground' }, 'Audit findings'),
          h('div', { className: 'flex gap-2 flex-wrap' },
            h('select', { className: 'rounded bg-black/40 px-2 py-1 text-xs border border-current/20', value: severityFilter, onChange: function (e) { setSeverityFilter(e.target.value); } }, ['all','critical','warning','suggestion'].map(function (v) { return h('option', { key: v, value: v }, 'severity: ' + v); })),
            h('select', { className: 'rounded bg-black/40 px-2 py-1 text-xs border border-current/20', value: sourceFilter, onChange: function (e) { setSourceFilter(e.target.value); } }, ['all','agentic','deterministic'].map(function (v) { return h('option', { key: v, value: v }, 'source: ' + v); }))
          )
        ),
        findingsSummaryView(findings),
        p.review_notes && p.review_notes.length && h('details', { className: 'rounded bg-black/20 p-2 text-xs' }, h('summary', { className: 'cursor-pointer font-semibold' }, 'Review notes'), h('ul', { className: 'list-disc pl-4 pt-2 space-y-1' }, listItems(p.review_notes))),
        sourceSection('Agentic findings', bySource.agentic),
        sourceSection('Deterministic findings', bySource.deterministic)
      );
    }
    function sectionActions(item) {
      return h('div', { className: 'flex gap-2 flex-wrap pt-1' },
        item.kind === 'opportunity' && h(ActionButton, { disabled: busy, onClick: function () { viewOpportunity(item.id); } }, 'Open opportunity'),
        item.kind === 'opportunity' && item.can_diagnose_ci && h(ActionButton, { disabled: busy, onClick: function () { fixCI(item); }, className: 'border-emerald-500/50' }, 'Fix CI'),
        item.kind === 'opportunity' && item.can_fix_issue && h(ActionButton, { disabled: busy, onClick: function () { fixIssue(item); }, className: 'border-emerald-500/50' }, 'Fix Issue'),
        item.workstream_id && h(ActionButton, { disabled: busy, onClick: function () { viewWorkstream(item.workstream_id); } }, 'Open workstream'),
        item.action_type === 'github_push_branch' && ['queued','edited_by_human'].includes(item.status) && h(ActionButton, { disabled: busy, onClick: function () { pushBranchNow(item); }, className: 'border-emerald-500/50' }, 'Push branch'),
        item.board_state !== 'in_progress' && h(ActionButton, { disabled: busy, onClick: function () { moveBoardState(item, 'in_progress'); } }, 'Move to In progress'),
        item.board_state !== 'in_review' && h(ActionButton, { disabled: busy, onClick: function () { moveBoardState(item, 'in_review'); } }, 'Move to Review'),
        item.board_state !== 'done' && h(ActionButton, { disabled: busy, onClick: function () { moveBoardState(item, 'done'); } }, 'Mark done'),
        item.board_state !== 'archived' && h(ActionButton, { disabled: busy, onClick: function () { moveBoardState(item, 'archived'); } }, 'Archive'),
        item.board_state === 'archived' && h(ActionButton, { disabled: busy, onClick: function () { moveBoardState(item, 'done'); } }, 'Unarchive')
      );
    }
    function sectionCard(title, items, emptyText) {
      return h(Card, null,
        h(CardHeader, null, h(CardTitle, null, title + ' (' + items.length + ')')),
        h(CardContent, { className: 'space-y-2' },
          !items.length && h('div', { className: 'text-xs text-text-secondary' }, emptyText),
          items.slice(0, 8).map(function (item) {
            return h('div', { key: item.kind + ':' + item.id, className: 'rounded bg-black/20 p-2 text-xs space-y-1' },
              h('div', { className: 'font-semibold' }, item.title || item.opportunity_title || item.summary),
              h('div', { className: 'flex gap-1 flex-wrap' }, item.repo && h(Badge, null, item.repo), item.workstream_stage && h(Badge, null, item.workstream_stage.replace(/_/g, ' ')), item.status && h(Badge, null, item.status)),
              item.current_step && h('div', { className: 'text-text-secondary' }, item.current_step),
              sectionActions(item)
            );
          })
        )
      );
    }
    function boardItemCard(item) {
      const title = item.title || item.opportunity_title || item.summary || item.id;
      const description = item.kind === 'opportunity' ? item.description : item.summary;
      return h('div', {
        key: item.kind + ':' + item.id,
        className: 'hermes-kanban-card visibility-os-card',
        'data-visibility-kind': item.kind,
        draggable: true,
        role: 'button',
        tabIndex: 0,
        onClick: function (e) {
          if (e.target && e.target.closest && e.target.closest('button, a, input, select, textarea, summary, details')) return;
          if (item.kind === 'opportunity') viewOpportunity(item.id);
          else setSelectedTicket(item);
        },
        onKeyDown: function (e) {
          if (e.key !== 'Enter' && e.key !== ' ') return;
          e.preventDefault();
          if (item.kind === 'opportunity') viewOpportunity(item.id);
          else setSelectedTicket(item);
        },
        onDragStart: function (e) {
          e.dataTransfer.setData('application/vnd.visibility-os-card', JSON.stringify({ kind: item.kind, id: item.id }));
          e.dataTransfer.effectAllowed = 'move';
        }
      },
        h('div', { className: 'hermes-kanban-card-content' },
        h('div', { className: 'hermes-kanban-card-row visibility-os-card-head' },
          h('div', { className: 'hermes-kanban-card-title' }, title),
          h(Badge, null, item.kind === 'opportunity' ? 'Opportunity' : 'Action')
        ),
        h('div', { className: 'hermes-kanban-card-row hermes-kanban-card-meta' },
          item.repo && h(Badge, null, item.repo),
          item.source_repo && h(Badge, null, item.source_repo),
          item.category && h(Badge, null, item.category),
          item.status && h(Badge, null, item.status),
          item.priority_score && h(Badge, null, 'P' + item.priority_score),
          item.board_state_actor && h(Badge, null, 'manual')
        ),
        description && h('p', { className: 'visibility-os-card-description' }, description),
        item.current_step && h('div', { className: 'visibility-os-card-step' }, item.current_step),
        item.workstream_id && workstreamBadge(item),
        item.action_type === 'github_push_branch' && proposedPRView(item),
        findingsView(item),
        sectionActions(item)
        )
      );
    }
    function kanbanColumn(key, title, emptyText) {
      const items = allItems.filter(function (i) { return (i.board_state || 'todo') === key; });
      const help = {
        todo: 'Unstarted opportunities and queued work.',
        in_progress: 'Agent or human work currently moving.',
        in_review: 'Proposed PRs and decisions waiting on review.',
        done: 'Completed, resolved, rejected, or failed work.',
        archived: 'Hidden history kept for auditability.'
      }[key] || '';
      function handleDrop(e) {
        e.preventDefault();
        let raw = e.dataTransfer.getData('application/vnd.visibility-os-card');
        if (!raw) return;
        try {
          const payload = JSON.parse(raw);
          const item = allItems.find(function (i) { return i.kind === payload.kind && i.id === payload.id; });
          if (item && item.board_state !== key) moveBoardState(item, key);
        } catch (_) {}
      }
      return h('div', {
        className: 'hermes-kanban-column visibility-os-column',
        'data-kanban-column': key,
        onDragOver: function (e) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; },
        onDrop: handleDrop
      },
        h('div', { className: 'hermes-kanban-column-header', title: help },
          h('span', { className: 'hermes-kanban-dot visibility-os-dot visibility-os-dot-' + key }),
          h('span', { className: 'hermes-kanban-column-label' }, title),
          h('span', { className: 'hermes-kanban-column-count' }, items.length)
        ),
        h('div', { className: 'hermes-kanban-column-sub' }, help),
        h('div', { className: 'hermes-kanban-column-body' },
          !items.length && h('div', { className: 'hermes-kanban-empty' }, emptyText),
          items.map(boardItemCard)
        )
      );
    }
    const allItems = feed.items || [];
    const boardCounts = (feed.counts && feed.counts.board) || {};
    return h('div', { className: 'hermes-kanban visibility-os-kanban h-full min-h-0 overflow-auto p-6 space-y-4' },
      h('div', { className: 'flex items-center justify-between gap-4' },
        h('div', null,
          h('h1', { className: 'text-2xl font-bold text-midground' }, 'Hermes Visibility OS'),
          h('p', { className: 'text-sm text-text-secondary' }, 'Evidence-backed opportunities and human-approved write actions.')
        ),
        h('div', { className: 'flex gap-2 flex-wrap justify-end' },
          h(ActionButton, { onClick: load, disabled: busy }, busy ? 'Loading…' : 'Refresh'),
          h(ActionButton, { onClick: function () { post('/api/plugins/visibility-os/scan/github/all', {}); }, disabled: busy }, 'Scan GitHub Repos'),
          h(ActionButton, { onClick: function () { post('/api/plugins/visibility-os/daily-plan', {}); }, disabled: busy }, 'Generate Daily Plan')
        )
      ),
      error && h('div', { className: 'rounded border border-red-500/40 p-3 text-red-300' }, error),
      selectedWorkstream && workstreamDetailView(selectedWorkstream),
      selectedTicket && ticketModalView(selectedTicket),
      selectedOpportunity && opportunityModalView(selectedOpportunity),
      h('div', { className: 'visibility-os-metrics' },
        h(Card, null, h(CardContent, { className: 'p-4' }, h('div', { className: 'text-2xl font-bold' }, boardCounts.todo || 0), h('div', { className: 'text-xs text-text-secondary' }, 'Todo'))),
        h(Card, null, h(CardContent, { className: 'p-4' }, h('div', { className: 'text-2xl font-bold' }, boardCounts.in_progress || 0), h('div', { className: 'text-xs text-text-secondary' }, 'In progress'))),
        h(Card, null, h(CardContent, { className: 'p-4' }, h('div', { className: 'text-2xl font-bold' }, boardCounts.in_review || 0), h('div', { className: 'text-xs text-text-secondary' }, 'In review'))),
        h(Card, null, h(CardContent, { className: 'p-4' }, h('div', { className: 'text-2xl font-bold' }, boardCounts.done || 0), h('div', { className: 'text-xs text-text-secondary' }, 'Done'))),
        h(Card, null, h(CardContent, { className: 'p-4' }, h('div', { className: 'text-2xl font-bold' }, boardCounts.archived || 0), h('div', { className: 'text-xs text-text-secondary' }, 'Archived')))
      ),
      h('div', { className: 'visibility-os-board-toolbar' },
        h('div', null,
          h('div', { className: 'text-sm font-bold text-midground' }, 'Kanban board'),
          h('div', { className: 'text-xs text-text-secondary' }, 'Move cards through Todo, In progress, In review, Done, then archive them out of the active board.')
        ),
        h('label', { className: 'inline-flex items-center gap-2 text-xs text-text-secondary' },
          h('input', { type: 'checkbox', checked: showArchived, onChange: function (e) { setShowArchived(e.target.checked); } }),
          'Show archived'
        )
      ),
      h('div', { className: 'hermes-kanban-columns visibility-os-kanban-columns' },
        kanbanColumn('todo', 'Todo', 'No unstarted opportunities.'),
        kanbanColumn('in_progress', 'In progress', 'No agent or human work in progress.'),
        kanbanColumn('in_review', 'In review', 'No proposed PRs or decisions waiting.'),
        kanbanColumn('done', 'Done', 'No completed work yet.'),
        showArchived ? kanbanColumn('archived', 'Archived', 'Nothing archived yet.') : null
      )
    );
  }
  plugins.register('visibility-os', VisibilityOS);
})();
