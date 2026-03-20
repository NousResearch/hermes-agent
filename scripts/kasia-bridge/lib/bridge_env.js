function parseListEnv(value) {
  return [
    ...new Set(
      String(value || "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean)
    ),
  ];
}

function parseBroadcastSubscriptionsEnv(value) {
  const parsed = {};
  for (const segment of String(value || "").split(";")) {
    const trimmed = segment.trim();
    if (!trimmed) {
      continue;
    }
    const [channelName, publishers] = trimmed.split("=", 2);
    if (!channelName || !publishers) {
      continue;
    }
    parsed[channelName.trim()] = publishers
      .split("|")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return parsed;
}

export function readBridgeEnv(env = process.env) {
  const seedPhrase = String(env.KASIA_SEED_PHRASE || "").trim();
  const singularIndexerUrl = String(env.KASIA_INDEXER_URL || "").trim();
  const singularNodeUrl = String(env.KASIA_NODE_WBORSH_URL || "").trim();
  const indexerUrls = parseListEnv(env.KASIA_INDEXER_URLS || "");
  const nodeUrls = parseListEnv(env.KASIA_NODE_WBORSH_URLS || "");

  const effectiveIndexerUrls = indexerUrls.length
    ? indexerUrls
    : singularIndexerUrl
      ? [singularIndexerUrl]
      : [];
  const effectiveNodeUrls = nodeUrls.length
    ? nodeUrls
    : singularNodeUrl
      ? [singularNodeUrl]
      : [];

  return {
    seedPhrase,
    indexerUrl: singularIndexerUrl || effectiveIndexerUrls[0] || "",
    nodeUrl: singularNodeUrl || effectiveNodeUrls[0] || "",
    indexerUrls: effectiveIndexerUrls,
    nodeUrls: effectiveNodeUrls,
    network: String(env.KASIA_NETWORK || "mainnet").trim() || "mainnet",
    knsUrl: String(env.KASIA_KNS_URL || "").trim(),
    feePolicy: String(env.KASIA_FEE_POLICY || "priority").trim() || "priority",
    maxMultipartParts: Number.parseInt(env.KASIA_MAX_MULTIPARTS || "8", 10),
    contextualMessageTargetChars: Number.parseInt(
      env.KASIA_TARGET_MESSAGE_CHARS || "240",
      10
    ),
    broadcastSubscriptions: parseBroadcastSubscriptionsEnv(
      env.KASIA_BROADCAST_SUBSCRIPTIONS || ""
    ),
    allowedBroadcastChannels: parseListEnv(
      env.KASIA_ALLOWED_BROADCAST_CHANNELS || ""
    ),
    allowAllBroadcastChannels: ["true", "1", "yes"].includes(
      String(env.KASIA_ALLOW_ALL_BROADCAST_CHANNELS || "").toLowerCase()
    ),
  };
}

export function validateBridgeEnv(config) {
  if (!config.seedPhrase || config.indexerUrls.length === 0 || config.nodeUrls.length === 0) {
    throw new Error(
      "KASIA_SEED_PHRASE plus either KASIA_INDEXER_URL or KASIA_INDEXER_URLS and either KASIA_NODE_WBORSH_URL or KASIA_NODE_WBORSH_URLS are required"
    );
  }
}

export { parseBroadcastSubscriptionsEnv, parseListEnv };
