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
function fail(message) {
  state.textContent = 'Disconnected';
  button.disabled = true;
  error.hidden = false;
  error.textContent = message;
  document.body.dataset.connected = 'false';
}
function connect() {
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
    document.body.dataset.connected = 'true';
    document.body.dataset.control = String(control);
    state.textContent = control ? 'Your control · live' : 'Live · view only';
    button.textContent = control ? 'Return to agent' : 'Take over';
    button.disabled = false;
  });
  connection.addEventListener('disconnect', () => {
    if (current === generation) fail('The realm stopped, this viewer expired, or the connection closed. Use Watch in Hermes to reconnect.');
  });
  connection.addEventListener('securityfailure', () => fail('Viewer authorization failed. Open a fresh Watch link from Hermes.'));
}
button.addEventListener('click', () => { control = !control; connect(); });
if (!token || !/^[A-Za-z0-9_-]{40,128}$/.test(token)) fail('Open this viewer with Watch in Hermes. A current realm capability is required.');
else connect();
