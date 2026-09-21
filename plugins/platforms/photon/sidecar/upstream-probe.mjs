// Authenticated unary probe used to distinguish a dead inbound stream from an offline channel.

import { classifyProbeRejection, createProbeMessageId } from "./stream-staleness.mjs";

export async function probeUpstream({ app, imessage, spaceId, timeoutMs, staleness }) {
  if (typeof app?.stop !== "function") {
    return { alive: false, hung: false, reason: "spectrum app not constructed" };
  }
  const probeId = createProbeMessageId();
  let timer = null;
  const timeout = new Promise((resolve) => {
    timer = setTimeout(
      () => resolve({ alive: false, hung: true, reason: "probe timed out" }),
      timeoutMs
    );
    timer.unref();
  });
  const attempt = (async () => {
    try {
      const im = imessage(app);
      const space = await im.space.get(spaceId);
      await space.getMessage(probeId);
      return { alive: true, hung: false, reason: "round-trip completed" };
    } catch (error) {
      const verdict = classifyProbeRejection(error);
      return { alive: verdict.alive, hung: false, reason: verdict.reason };
    }
  })();
  const outcome = await Promise.race([attempt, timeout]);
  if (timer) clearTimeout(timer);
  staleness.lastProbeAt = Date.now();
  staleness.lastProbeOutcome = outcome.alive ? "alive" : "inconclusive";
  return outcome;
}
