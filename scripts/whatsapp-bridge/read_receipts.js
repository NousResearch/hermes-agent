export async function sendReadReceipt(sock, msg, log = console) {
  const key = msg?.key;
  if (!sock?.readMessages || !key?.id || !key?.remoteJid || key.fromMe) {
    return false;
  }

  try {
    await sock.readMessages([key]);
    return true;
  } catch (err) {
    log.error?.('[bridge] Failed to send read receipt:', err?.message || err);
    return false;
  }
}
