import { proto } from '@whiskeysockets/baileys';
import { makeShortcakeFlow } from './passkey_flow.js';

// Baileys still owns acknowledgement of these notifications. The handshake
// runs separately so an interactive browser prompt cannot delay that ACK.
export function attachPasskeyFlow(socket, { logger, signAssertion }) {
  let closed = false;
  let pending = Promise.resolve();
  const flow = signAssertion ? makeShortcakeFlow({
    logger,
    query: node => socket.query(node),
    signAssertion,
    getCreds: () => socket.authState.creds,
    updateCreds: patch => socket.ev.emit('creds.update', patch),
    deviceType: proto.DeviceProps.PlatformType.CHROME,
    emitVerificationCode: (code, skipHandoffUX) => socket.ev.emit('connection.update', {
      passkeyVerification: { code, skipHandoffUX },
    }),
  }) : null;

  function notification(node) {
    if (closed || !['s.whatsapp.net', '@s.whatsapp.net'].includes(node.attrs.from)) return;
    if (!['passkey_prologue_request', 'crsc_continuation'].includes(node.attrs.type)) return;
    if (node.attrs.type === 'passkey_prologue_request') {
      socket.ev.emit('connection.update', { passkeyRequired: { hasSigner: Boolean(flow) } });
    }
    if (!flow) return;
    pending = pending.then(() => {
      if (!closed) return flow.handleIncomingNotification(node);
    }).catch(error => {
      if (closed || error?.name === 'AbortError') return;
      flow.clearSession();
      socket.ev.emit('connection.update', { passkeyError: error });
    });
  }

  function connection(update) {
    if (update.connection !== 'close') return;
    closed = true;
    flow?.clearSession();
    socket.ws.off('CB:notification', notification);
    socket.ev.off('connection.update', connection);
  }

  socket.ws.on('CB:notification', notification);
  socket.ev.on('connection.update', connection);
  return {
    confirmVerificationCode() {
      if (!flow || closed) throw new Error('No active passkey pairing');
      return flow.confirmVerificationCode();
    },
  };
}
