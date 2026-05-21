(function () {
  const SDK = window.__HERMES_PLUGIN_SDK__;
  if (!SDK || !window.__HERMES_PLUGINS__) return;

  const React = SDK.React;
  const h = React.createElement;
  const useState = SDK.hooks.useState;
  const useEffect = SDK.hooks.useEffect;
  const C = SDK.components || {};
  const Card = C.Card || 'div';
  const CardHeader = C.CardHeader || 'div';
  const CardTitle = C.CardTitle || 'h2';
  const CardContent = C.CardContent || 'div';
  const Button = C.Button || 'button';
  const Input = C.Input || 'input';
  const Label = C.Label || 'label';
  const Badge = C.Badge || 'span';
  const Separator = C.Separator || 'hr';
  const fetchJSON = SDK.fetchJSON;

  const api = {
    summary: () => fetchJSON('/api/plugins/hermes-git-projects/summary'),
    importRepo: (payload) => fetchJSON('/api/plugins/hermes-git-projects/import', { method: 'POST', body: JSON.stringify(payload) }),
    source: (id, payload) => fetchJSON('/api/plugins/hermes-git-projects/projects/' + encodeURIComponent(id) + '/source-control', { method: 'POST', body: JSON.stringify(payload) }),
    issue: (id, payload) => fetchJSON('/api/plugins/hermes-git-projects/projects/' + encodeURIComponent(id) + '/issues', { method: 'POST', body: JSON.stringify(payload) }),
    saveSkills: (skills) => fetchJSON('/api/plugins/hermes-git-projects/suggested-skills', { method: 'PUT', body: JSON.stringify({ skills }) }),
  };

  function cx() { return Array.from(arguments).filter(Boolean).join(' '); }
  function small(text, className) { return h('div', { className: cx('text-xs text-muted-foreground', className) }, text); }
  function badge(text, variant) { return h(Badge, { variant: variant || 'secondary', className: 'mr-1 mb-1' }, text); }

  function GitProjectsPage() {
    const [loading, setLoading] = useState(true);
    const [busy, setBusy] = useState('');
    const [error, setError] = useState('');
    const [data, setData] = useState(null);
    const [selectedId, setSelectedId] = useState('');
    const [repoUrl, setRepoUrl] = useState('');
    const [importBranch, setImportBranch] = useState('');
    const [branchName, setBranchName] = useState('');
    const [issue, setIssue] = useState({ title: '', body: '', kind: 'bug', severity: 'medium', labels: '', parent_task_ids: '', selected_skills: [] });
    const [skillDraft, setSkillDraft] = useState(null);

    function refresh(selectId) {
      setLoading(true);
      setError('');
      return api.summary().then((payload) => {
        setData(payload);
        setSkillDraft(payload.suggested_skills || []);
        if (selectId) setSelectedId(selectId);
        else if (!selectedId && payload.projects && payload.projects[0]) setSelectedId(payload.projects[0].id);
      }).catch((err) => setError(String(err && err.message || err))).finally(() => setLoading(false));
    }

    useEffect(() => { refresh(); }, []);

    const projects = (data && data.projects) || [];
    const selected = projects.find((p) => p.id === selectedId) || projects[0];
    const skills = skillDraft || [];

    function run(label, fn) {
      setBusy(label);
      setError('');
      return fn().catch((err) => setError(String(err && err.message || err))).finally(() => setBusy(''));
    }

    function importRepo(e) {
      e.preventDefault();
      if (!repoUrl.trim()) return;
      run('Importing repository...', () => api.importRepo({ repo_url: repoUrl.trim(), branch: importBranch.trim() || null }).then((payload) => {
        setRepoUrl('');
        setImportBranch('');
        return refresh(payload.project && payload.project.id);
      }));
    }

    function sourceAction(action) {
      if (!selected) return;
      const payload = { action, branch: (action === 'checkout' || action === 'create_branch') ? branchName.trim() : null };
      run(action + '...', () => api.source(selected.id, payload).then((payload) => {
        if (payload.project) setSelectedId(payload.project.id);
        return refresh(selected.id);
      }));
    }

    function toggleSkill(name) {
      const current = issue.selected_skills || [];
      const next = current.includes(name) ? current.filter((s) => s !== name) : current.concat([name]);
      setIssue(Object.assign({}, issue, { selected_skills: next }));
    }

    function createIssue(e) {
      e.preventDefault();
      if (!selected || !issue.title.trim()) return;
      const labels = issue.labels.split(',').map((s) => s.trim()).filter(Boolean);
      const parents = issue.parent_task_ids.split(',').map((s) => s.trim()).filter(Boolean);
      const selectedSkills = (issue.selected_skills && issue.selected_skills.length) ? issue.selected_skills : null;
      run('Creating issue todo...', () => api.issue(selected.id, {
        title: issue.title,
        body: issue.body,
        kind: issue.kind,
        severity: issue.severity,
        labels,
        parent_task_ids: parents,
        selected_skills: selectedSkills,
      }).then(() => {
        setIssue({ title: '', body: '', kind: 'bug', severity: 'medium', labels: '', parent_task_ids: '', selected_skills: [] });
        return refresh(selected.id);
      }));
    }

    function updateSkill(index, field, value) {
      const next = skills.slice();
      next[index] = Object.assign({}, next[index], { [field]: value });
      setSkillDraft(next);
    }

    function saveSkills() {
      const cleaned = skills.map((s) => Object.assign({}, s, {
        triggers: Array.isArray(s.triggers) ? s.triggers : String(s.triggers || '').split(',').map((x) => x.trim()).filter(Boolean)
      })).filter((s) => s.name && s.name.trim());
      run('Saving skills...', () => api.saveSkills(cleaned).then(() => refresh(selected && selected.id)));
    }

    return h('div', { className: 'space-y-6' },
      h('div', { className: 'flex flex-col gap-2 md:flex-row md:items-end md:justify-between' },
        h('div', null,
          h('h1', { className: 'text-2xl font-semibold tracking-tight' }, 'Hermes Git Projects'),
          h('p', { className: 'text-sm text-muted-foreground max-w-3xl' }, 'Import Git repository URLs into managed local clones, scan them as Hermes-ready projects, log issues as Kanban todos, choose suggested skills, and run safe source-control actions from the dashboard.')
        ),
        data && h('div', { className: 'text-xs text-muted-foreground text-right' },
          h('div', null, 'Storage: ', data.storage && data.storage.base),
          h('div', null, projects.length + ' project(s) scanned')
        )
      ),

      error && h(Card, { className: 'border-destructive' }, h(CardContent, { className: 'py-3 text-sm text-destructive' }, error)),
      busy && h(Card, null, h(CardContent, { className: 'py-3 text-sm' }, busy)),

      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, 'Import repository')),
        h(CardContent, null,
          h('form', { onSubmit: importRepo, className: 'grid gap-3 md:grid-cols-[1fr_180px_auto]' },
            h('div', null,
              h(Label, null, 'Repository URL'),
              h(Input, { value: repoUrl, onChange: (e) => setRepoUrl(e.target.value), placeholder: 'https://github.com/org/repo.git or git@github.com:org/repo.git' })
            ),
            h('div', null,
              h(Label, null, 'Branch optional'),
              h(Input, { value: importBranch, onChange: (e) => setImportBranch(e.target.value), placeholder: 'main' })
            ),
            h(Button, { type: 'submit', disabled: !!busy || !repoUrl.trim(), className: 'self-end' }, 'Import')
          )
        )
      ),

      h('div', { className: 'grid gap-4 lg:grid-cols-[360px_1fr]' },
        h(Card, null,
          h(CardHeader, null, h(CardTitle, null, 'Projects')),
          h(CardContent, { className: 'space-y-2' },
            loading ? small('Scanning projects...') : projects.length ? projects.map((p) => h('button', {
              key: p.id,
              onClick: () => setSelectedId(p.id),
              className: cx('w-full rounded-md border p-3 text-left transition hover:bg-muted', selected && selected.id === p.id && 'border-primary bg-muted')
            },
              h('div', { className: 'flex items-center justify-between gap-2' },
                h('div', { className: 'font-medium truncate' }, p.name),
                p.ready ? badge('ready') : badge('error', 'destructive')
              ),
              small(p.path, 'truncate'),
              p.source_control && h('div', { className: 'mt-2 flex flex-wrap gap-1 text-xs' },
                badge(p.source_control.branch || 'detached'),
                p.source_control.dirty ? badge('dirty', 'destructive') : badge('clean'),
                p.issue_count ? badge(p.issue_count + ' issues') : null
              )
            )) : small('No projects yet. Import a repo URL above.')
          )
        ),

        selected ? h('div', { className: 'space-y-4' },
          h(Card, null,
            h(CardHeader, null, h(CardTitle, null, selected.name)),
            h(CardContent, { className: 'space-y-3' },
              small(selected.description || 'Ready for Hermes work.'),
              h('div', { className: 'flex flex-wrap gap-1' }, (selected.stack || []).map((s) => badge(s))),
              h(Separator, null),
              h('div', { className: 'grid gap-2 md:grid-cols-2 text-sm' },
                h('div', null, h('strong', null, 'Branch: '), selected.source_control && selected.source_control.branch),
                h('div', null, h('strong', null, 'Upstream: '), selected.source_control && (selected.source_control.upstream || 'none')),
                h('div', null, h('strong', null, 'Ahead/behind: '), selected.source_control ? (selected.source_control.ahead + '/' + selected.source_control.behind) : '0/0'),
                h('div', null, h('strong', null, 'Latest: '), selected.source_control && selected.source_control.latest_commit),
                h('div', { className: 'md:col-span-2 truncate' }, h('strong', null, 'Remote: '), selected.source_control && selected.source_control.remote)
              ),
              h('div', { className: 'flex flex-wrap gap-2' },
                h(Button, { type: 'button', variant: 'secondary', onClick: () => sourceAction('fetch'), disabled: !!busy }, 'Fetch'),
                h(Button, { type: 'button', variant: 'secondary', onClick: () => sourceAction('pull'), disabled: !!busy }, 'Pull ff-only'),
                h(Button, { type: 'button', variant: 'secondary', onClick: () => sourceAction('push'), disabled: !!busy }, 'Push current'),
                h(Input, { className: 'w-56', value: branchName, onChange: (e) => setBranchName(e.target.value), placeholder: 'branch name' }),
                h(Button, { type: 'button', variant: 'secondary', onClick: () => sourceAction('checkout'), disabled: !!busy || !branchName.trim() }, 'Checkout'),
                h(Button, { type: 'button', onClick: () => sourceAction('create_branch'), disabled: !!busy || !branchName.trim() }, 'Create branch')
              )
            )
          ),

          h(Card, null,
            h(CardHeader, null, h(CardTitle, null, 'Log issue → create Kanban todo')),
            h(CardContent, null,
              h('form', { onSubmit: createIssue, className: 'space-y-3' },
                h('div', { className: 'grid gap-3 md:grid-cols-3' },
                  h('div', { className: 'md:col-span-2' }, h(Label, null, 'Title'), h(Input, { value: issue.title, onChange: (e) => setIssue(Object.assign({}, issue, { title: e.target.value })), placeholder: 'Fix login redirect bug' })),
                  h('div', null, h(Label, null, 'Kind'), h(Input, { value: issue.kind, onChange: (e) => setIssue(Object.assign({}, issue, { kind: e.target.value })) }))
                ),
                h('div', null, h(Label, null, 'Details'), h('textarea', { className: 'min-h-24 w-full rounded-md border bg-background px-3 py-2 text-sm', value: issue.body, onChange: (e) => setIssue(Object.assign({}, issue, { body: e.target.value })), placeholder: 'What is broken, expected behavior, context, links...' })),
                h('div', { className: 'grid gap-3 md:grid-cols-3' },
                  h('div', null, h(Label, null, 'Severity'), h(Input, { value: issue.severity, onChange: (e) => setIssue(Object.assign({}, issue, { severity: e.target.value })) })),
                  h('div', null, h(Label, null, 'Labels comma-separated'), h(Input, { value: issue.labels, onChange: (e) => setIssue(Object.assign({}, issue, { labels: e.target.value })) })),
                  h('div', null, h(Label, null, 'Parent task ids comma-separated'), h(Input, { value: issue.parent_task_ids, onChange: (e) => setIssue(Object.assign({}, issue, { parent_task_ids: e.target.value })) }))
                ),
                h('div', null,
                  h(Label, null, 'Suggested skills for this issue'),
                  h('div', { className: 'mt-2 grid gap-2 md:grid-cols-2' }, skills.map((s) => h('label', { key: s.name, className: 'flex gap-2 rounded-md border p-2 text-sm' },
                    h('input', { type: 'checkbox', checked: (issue.selected_skills || []).includes(s.name), onChange: () => toggleSkill(s.name) }),
                    h('span', null, h('span', { className: 'font-medium' }, s.label || s.name), small(s.reason || s.name))
                  )))
                ),
                h(Button, { type: 'submit', disabled: !!busy || !issue.title.trim() }, 'Save issue and create todo')
              )
            )
          ),

          h(Card, null,
            h(CardHeader, null, h(CardTitle, null, 'Recent project issues')),
            h(CardContent, { className: 'space-y-2' },
              (selected.issues || []).length ? selected.issues.map((it) => h('div', { key: it.id, className: 'rounded-md border p-3' },
                h('div', { className: 'flex items-center justify-between gap-2' }, h('div', { className: 'font-medium' }, it.title), it.todo_id ? badge('todo ' + it.todo_id) : badge('todo pending/error', 'destructive')),
                small((it.kind || 'issue') + ' • ' + (it.severity || 'medium') + ' • branch ' + (it.recommended_branch || '')),
                it.selected_skills && h('div', { className: 'mt-2' }, it.selected_skills.map((s) => badge(s)))
              )) : small('No issues logged yet.')
            )
          )
        ) : h(Card, null, h(CardContent, { className: 'py-8' }, small('Select or import a project.')))
      ),

      h(Card, null,
        h(CardHeader, null, h(CardTitle, null, 'Suggested skills storage')),
        h(CardContent, { className: 'space-y-3' },
          small('These are stored in the plugin profile state. Users can choose them per issue; defaults are auto-attached when no explicit selection is made.'),
          skills.map((s, i) => h('div', { key: i, className: 'grid gap-2 rounded-md border p-3 md:grid-cols-[220px_1fr_120px]' },
            h(Input, { value: s.name || '', onChange: (e) => updateSkill(i, 'name', e.target.value), placeholder: 'skill-name' }),
            h(Input, { value: s.reason || '', onChange: (e) => updateSkill(i, 'reason', e.target.value), placeholder: 'why this skill is useful' }),
            h('label', { className: 'flex items-center gap-2 text-sm' }, h('input', { type: 'checkbox', checked: !!s.default, onChange: (e) => updateSkill(i, 'default', e.target.checked) }), 'Default')
          )),
          h('div', { className: 'flex gap-2' },
            h(Button, { type: 'button', variant: 'secondary', onClick: () => setSkillDraft(skills.concat([{ name: '', label: '', reason: '', default: false, triggers: [] }])) }, 'Add skill'),
            h(Button, { type: 'button', onClick: saveSkills, disabled: !!busy }, 'Save suggested skills')
          )
        )
      )
    );
  }

  window.__HERMES_PLUGINS__.register('hermes-git-projects', GitProjectsPage);
})();
