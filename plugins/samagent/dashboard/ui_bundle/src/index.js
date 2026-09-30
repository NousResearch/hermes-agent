/**
 * Mission Control root.
 *
 * Owns exactly three things: the polled backend snapshot, the action wrappers
 * that POST to it, and theme. Everything else is derived in data.js or local to
 * a stage. Keeping state this narrow is what makes the rest of the tree
 * stateless enough to reason about.
 */
import { createElement as h, useState, useEffect, useCallback, useRef } from './sdk.js';
import { apiGet, apiPost } from './data.js';
import { MissionControl } from './app.js';
import { registerRoot } from './sdk.js';

const POLL_MS = 4000;

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try {
      const stored = localStorage.getItem('samagent.theme');
      if (stored === 'dark' || stored === 'light') return stored;
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    } catch (e) {
      return 'light';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('samagent.theme', theme);
    } catch (e) {
      /* private mode */
    }
  }, [theme]);

  return [theme, useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), [])];
}

function Root() {
  const [state, setState] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [theme, toggleTheme] = useTheme();
  const noticeTimer = useRef(null);

  /* ---- polling ---------------------------------------------------------
     A poll is a cache refresh, never a clobber: it replaces the snapshot but
     never resets what the user is looking at. A failed poll keeps the last
     good snapshot on screen and surfaces a degraded hint rather than blanking
     the app — one of the distinct states a loading UI owes the user. */
  useEffect(() => {
    let cancelled = false;

    const refresh = () =>
      apiGet('/state')
        .then((data) => {
          if (cancelled) return;
          setState(data);
          setError(null);
        })
        .catch((err) => {
          if (cancelled) return;
          setError(err && err.message ? err.message : String(err));
        });

    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const flash = useCallback((message) => {
    setNotice(message);
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(null), 6000);
  }, []);

  useEffect(() => () => noticeTimer.current && clearTimeout(noticeTimer.current), []);

  /* Every mutating call follows the same shape: mark busy, POST, fold the
     returned snapshot into state (the endpoints return the full state, so
     there is no refetch race), clear busy, surface failures without losing the
     view. */
  const act = useCallback(
    (fn, successMessage) =>
      async (...args) => {
        setBusy(true);
        try {
          const res = await fn(...args);
          if (res && typeof res === 'object' && (res.workspace || res.pre_prod_gate || res.todos)) {
            setState(res);
          } else {
            const fresh = await apiGet('/state');
            setState(fresh);
          }
          setError(null);
          if (successMessage) flash(successMessage);
          return res;
        } catch (err) {
          setError(err && err.message ? err.message : String(err));
          throw err;
        } finally {
          setBusy(false);
        }
      },
    [flash]
  );

  const actions = {
    interview: act((brief) => apiPost('/interview', { brief })),
    plan: act((brief, answers) => apiPost('/plan', { brief, answers: answers || {} })),
    build: act(({ autonomy, routerPolicy }) => apiPost('/build', { autonomy, router_policy: routerPolicy }), 'Building — the run view will fill in shortly'),
    steer: act((note) => apiPost('/steer', { action: 'steer', note }), 'Note sent to the running task'),
    reverify: act(() => apiPost('/reverify', {}), 'Re-verification finished'),
    promote: act(() => apiPost('/promote-prod', {}), 'Promoted to production'),
    createPr: act(() => apiPost('/github/pr', {}), 'Pull request opened'),
    syncGithub: act(() => apiPost('/github/sync', { commit_message: 'chore: samagent run', push_to_remote: true }), 'Committed and pushed'),
    openInVscode: act((relPath) => apiPost('/ide/open', { rel_path: relPath || null, line: 1 })),
    addFact: act((fact) => apiPost('/ledger/fact', fact), 'Saved to the ledger'),
    toggleTodo: act((todoId) => apiPost('/todo/action', { action: 'toggle', todo_id: todoId })),
    goStage: (id) => window.dispatchEvent(new CustomEvent('samagent:stage', { detail: id })),
  };

  if (!state) {
    return h(
      'div',
      { className: 'mc-boot' },
      error ? h('p', { className: 'mc-error' }, `Could not reach the Mission Control API — ${error}`) : h('p', { className: 'mc-muted' }, 'Loading the run…')
    );
  }

  return h(MissionControl, {
    state,
    actions,
    theme,
    onToggleTheme: toggleTheme,
    busy,
    notice: error ? `Degraded: ${error}` : notice,
  });
}

registerRoot(Root);
export default Root;
