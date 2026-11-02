import { strict as assert } from 'node:assert';
import express from 'express';

import { registerGroupRoutes } from './group_routes.js';

async function requestGroups({ socket, connected = true }) {
  const app = express();
  registerGroupRoutes(app, {
    getSocket: () => socket,
    isConnected: () => connected,
  });
  const server = app.listen(0, '127.0.0.1');
  await new Promise((resolve, reject) => {
    server.once('listening', resolve);
    server.once('error', reject);
  });
  try {
    const { port } = server.address();
    return await fetch(`http://127.0.0.1:${port}/groups`);
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
}

{
  let fetchCalls = 0;
  const response = await requestGroups({
    socket: {
      groupFetchAllParticipating: async () => {
        fetchCalls += 1;
        return {
          '120363001234567890@g.us': {
            id: '120363001234567890@g.us',
            subject: 'Hermes Operators',
            participants: [{ id: 'one' }, { id: 'two' }],
          },
          '120363009876543210@g.us': {
            subject: 'Release Room',
            participants: [],
          },
        };
      },
    },
  });
  assert.equal(response.status, 200);
  assert.equal(fetchCalls, 1);
  assert.deepEqual(await response.json(), [
    { id: '120363001234567890@g.us', name: 'Hermes Operators', participants: 2 },
    { id: '120363009876543210@g.us', name: 'Release Room', participants: 0 },
  ]);
  console.log('  ✓ GET /groups exposes participating group summaries');
}

{
  const response = await requestGroups({ socket: null, connected: false });
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), { error: 'Not connected to WhatsApp' });
  console.log('  ✓ GET /groups rejects requests while disconnected');
}

{
  const response = await requestGroups({
    socket: {
      groupFetchAllParticipating: async () => {
        throw new Error('group query failed');
      },
    },
  });
  assert.equal(response.status, 500);
  assert.deepEqual(await response.json(), { error: 'group query failed' });
  console.log('  ✓ GET /groups reports Baileys query failures');
}

console.log('\n✅ All WhatsApp group route tests passed.');
