// Quick WhatsApp QR - ESM version with proper pino logger
import path from 'path';
import fs from 'fs';
import { makeWASocket, useMultiFileAuthState, fetchLatestBaileysVersion, DisconnectReason } from '@whiskeysockets/baileys';
import QRCode from 'qrcode';
import { Boom } from '@hapi/boom';
import pino from 'pino';

const logger = pino({ level: 'warn' });
const SESSION_DIR = path.join(process.env.HOME || process.env.USERPROFILE, '.hermes', 'whatsapp', 'session');
const QR_OUTPUT = path.join(process.env.HOME || process.env.USERPROFILE, 'whatsapp-qr.png');

async function start() {
  const { state, saveCreds } = await useMultiFileAuthState(SESSION_DIR);
  const { version } = await fetchLatestBaileysVersion();
  const sock = makeWASocket({
    version, auth: state, logger,
    printQRInTerminal: false,
    browser: ['Hermes Agent', 'Chrome', '120.0'],
    syncFullHistory: false,
    markOnlineOnConnect: false,
    getMessage: async () => ({ conversation: '' }),
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', async (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      console.log('QR_RECEIVED');
      try {
        await QRCode.toFile(QR_OUTPUT, qr, { type: 'png', width: 500, margin: 2 });
        console.log('QR_SAVED:' + QR_OUTPUT);
        console.log('QR_DATA:' + qr);
      } catch(e) {
        console.log('QR_ERROR:' + e.message);
      }
    }
    if (connection === 'close') {
      const reason = new Boom(lastDisconnect?.error)?.output?.statusCode;
      if (reason === DisconnectReason.loggedOut) {
        console.log('LOGGED_OUT');
        process.exit(1);
      } else {
        console.log('RECONNECT');
        setTimeout(start, reason === 515 ? 1000 : 3000);
      }
    } else if (connection === 'open') {
      console.log('CONNECTED');
    }
  });
}

start();
