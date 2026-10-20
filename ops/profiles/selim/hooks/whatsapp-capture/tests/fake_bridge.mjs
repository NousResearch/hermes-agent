// A bridge stand-in for the capture-hook tests: the REAL ack watch and the same
// /health + /sends answers as bridge.js, with a fake socket the test drives.
//   POST /_sent?id=&chatId=   sock.sendMessage resolved for this message
//   POST /_ack?id=            WhatsApp's server acked it
import http from 'node:http';
import { EventEmitter } from 'node:events';
import { bridgeStatus, createAckWatch } from '../../../../../../scripts/whatsapp-bridge/ack_watch.js';

const watch = createAckWatch({ staleMs: Number(process.env.STALE_MS || 120_000) });
const sock = { ws: new EventEmitter(), ev: new EventEmitter() };
watch.attach(sock);

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://127.0.0.1');
  const q = Object.fromEntries(url.searchParams);
  let body;
  if (url.pathname === '/_sent') { watch.sent(q.id, q.chatId); body = { ok: true }; }
  else if (url.pathname === '/_ack') { sock.ws.emit('CB:ack,class:message', { attrs: { id: q.id } }); body = { ok: true }; }
  else if (url.pathname === '/sends') body = { chatId: q.chatId, ...watch.sends(q.chatId, Number(q.since || 0) * 1000) };
  else if (url.pathname === '/health') {
    const { stale, ...acks } = watch.health();
    body = { status: bridgeStatus('connected', { stale }), ...acks };
  } else { res.writeHead(404).end(); return; }
  res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify(body));
});
server.listen(0, '127.0.0.1', () => console.log(server.address().port));
