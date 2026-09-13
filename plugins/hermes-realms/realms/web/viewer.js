import RFB from './vendor/novnc/core/rfb.js';

const screen = document.querySelector('#screen');
const state = document.querySelector('#state');
const button = document.querySelector('#control');
const error = document.querySelector('#error');
const realm = location.pathname.split('/')[2];
const token = new URLSearchParams(location.hash.slice(1)).get('ticket');
// Capabilities never remain in history, requests, Referer, or browser storage.
history.replaceState(null, '', location.pathname);
let connection;
let generation = 0;
let control = false;
let retries = 0;
let retryTimer;
let stopped = false;
function fail(message) {
  clearTimeout(retryTimer);
  state.textContent = 'Disconnected';
  button.disabled = true;
  error.hidden = false;
  error.textContent = message;
  document.body.dataset.connected = 'false';
}
function connect() {
  if (stopped) return;
  clearTimeout(retryTimer);
  const current = ++generation;
  if (connection) connection.disconnect();
  button.disabled = true;
  state.textContent = 'Connecting…';
  error.hidden = true;
  const url = new URL(`/api/realms/${encodeURIComponent(realm)}/vnc`, location.origin);
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  if (control) url.searchParams.set('control', '1');
  connection = new RFB(screen, url.href, { wsProtocols: ['binary', `realm.${token}`] });
  connection.viewOnly = !control;
  connection.scaleViewport = true;
  connection.resizeSession = false;
  connection.showDotCursor = true;
  connection.addEventListener('connect', () => {
    if (current !== generation) return;
    retries = 0;
    document.body.dataset.connected = 'true';
    document.body.dataset.control = String(control);
    state.textContent = control ? 'Your control · live' : 'Live · view only';
    button.textContent = control ? 'Return to agent' : 'Take over';
    button.disabled = false;
  });
  connection.addEventListener('disconnect', () => {
    if (current !== generation || stopped) return;
    // A lost takeover lease must never be reclaimed without another click.
    control = false;
    document.body.dataset.connected = 'false';
    document.body.dataset.control = 'false';
    button.disabled = true;
    if (retries >= 5) {
      fail('The realm stopped or viewer authorization expired. Use Watch in Hermes to reconnect.');
      return;
    }
    state.textContent = 'Reconnecting…';
    retryTimer = setTimeout(connect, 1000 * 2 ** retries++);
  });
  connection.addEventListener('securityfailure', () => {
    if (current !== generation) return;
    stopped = true;
    fail('Viewer authorization failed. Open a fresh Watch link from Hermes.');
  });
}
window.addEventListener('pagehide', () => {
  stopped = true;
  generation++;
  clearTimeout(retryTimer);
  connection?.disconnect();
});
button.addEventListener('click', () => { control = !control; connect(); });
if (!token || !/^[A-Za-z0-9_-]{40,128}$/.test(token)) fail('Open this viewer with Watch in Hermes. A current realm capability is required.');
else connect();
