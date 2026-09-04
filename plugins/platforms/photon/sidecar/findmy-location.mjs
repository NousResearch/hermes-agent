const FIND_MY_BUNDLE = "com.apple.findmy.FindMyMessagesApp";

export function isFindMyLocationMessage(message) {
  const bundle = String(message?.balloonBundleId || "");
  const raw = message?.content?.raw;
  return (
    message?.content?.type === "custom" &&
    raw?.imessage_type === "unsupported-message" &&
    bundle.includes(FIND_MY_BUNDLE)
  );
}

export function normalizeLocationSnapshot(location) {
  if (!location) return null;
  const latitude = Number(location.latitude);
  const longitude = Number(location.longitude);
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  return {
    type: "location",
    latitude,
    longitude,
    accuracy: Number.isFinite(Number(location.accuracy)) ? Number(location.accuracy) : null,
    name: location.name || "",
    address: location.address || "",
    shortAddress: location.shortAddress || "",
    longAddress: location.longAddress || "",
    locationType: location.locationType || "",
  };
}

function chooseRoute(tokenData, phone) {
  if (tokenData?.type === "shared") {
    return {
      address: process.env.SPECTRUM_IMESSAGE_ADDRESS || "imessage.spectrum.photon.codes:443",
      token: tokenData.token,
    };
  }
  const auth = tokenData?.auth || {};
  const numbers = tokenData?.numbers || {};
  let instanceId = Object.keys(auth).find((id) => numbers[id] === phone);
  if (!instanceId && Object.keys(auth).length === 1) instanceId = Object.keys(auth)[0];
  if (!instanceId || !auth[instanceId]) return null;
  return { address: `${instanceId}.imsg.photon.codes:443`, token: auth[instanceId] };
}

export async function resolveFindMyLocation({
  message,
  projectId,
  projectSecret,
  issueTokens,
  createClient,
}) {
  if (!isFindMyLocationMessage(message)) return null;
  const sender = message?.sender?.address || message?.sender?.id || "";
  if (!sender || !projectId || !projectSecret) return null;
  const tokenData = await issueTokens(projectId, projectSecret);
  const route = chooseRoute(tokenData, message?.space?.phone);
  if (!route) return null;
  const client = createClient({
    address: route.address,
    tls: true,
    token: route.token,
    retry: true,
    autoIdempotency: true,
  });
  try {
    return normalizeLocationSnapshot(await client.locations.get(sender));
  } finally {
    await client.close();
  }
}
