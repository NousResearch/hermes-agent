export function participatingGroupSummaries(groups) {
  if (!groups || typeof groups !== 'object') return [];
  return Object.entries(groups).map(([jid, metadata]) => ({
    id: String(metadata?.id || jid),
    name: String(metadata?.subject || ''),
    participants: Array.isArray(metadata?.participants) ? metadata.participants.length : 0,
  }));
}

export function registerGroupRoutes(app, { getSocket, isConnected }) {
  app.get('/groups', async (req, res) => {
    const socket = getSocket();
    if (!socket || !isConnected()) {
      return res.status(503).json({ error: 'Not connected to WhatsApp' });
    }

    try {
      const groups = await socket.groupFetchAllParticipating();
      return res.json(participatingGroupSummaries(groups));
    } catch (err) {
      return res.status(500).json({ error: err.message });
    }
  });
}
