import { makeWASocket, useMultiFileAuthState, DisconnectReason, fetchLatestBaileysVersion } from '@whiskeysockets/baileys';
import pino from 'pino';
import { Boom } from '@hapi/boom';
import { writeFileSync, mkdirSync } from 'fs';
import { dirname } from 'path';

const SESSION_DIR = process.argv[2] || `${process.env.HOME}/.hermes/whatsapp/session`;
const QR_FILE = process.argv[3] || `${process.env.HOME}/.hermes/whatsapp/latest_qr.txt`;
mkdirSync(dirname(QR_FILE), { recursive: true });

console.log(`pair-export session=${SESSION_DIR}`);

async function startSocket() {
  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
  const { version } = await fetchLatestBaileysVersion();
  const sock = makeWASocket({
    version,
    auth: state,
    logger: pino({ level: 'silent' }),
    browser: ['Hermes Agent', 'Chrome', '1.0.0'],
  });

  sock.ev.on('creds.update', saveCreds);
  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      writeFileSync(QR_FILE, qr, 'utf8');
      console.log(`QR_FILE=${QR_FILE}`);
    }
    if (connection === 'open') {
      console.log('CONNECTED');
      process.exit(0);
    }
    if (connection === 'close') {
      const reason = new Boom(lastDisconnect?.error)?.output?.statusCode;
      console.log(`CLOSED reason=${reason || 'unknown'}`);
      if (reason === DisconnectReason.restartRequired || reason === 515) {
        console.log('RESTARTING_AFTER_515');
        setTimeout(() => startSocket().catch(err => { console.error(err); process.exit(1); }), 1000);
      } else {
        process.exit(1);
      }
    }
  });
}

startSocket().catch(err => { console.error(err); process.exit(1); });
