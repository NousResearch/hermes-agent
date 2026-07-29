export function buildStreamHealthSnapshot(
  health,
  restartAfterMs,
  now = Date.now(),
) {
  const degradedForMs =
    health.degradedSince === null ? 0 : now - health.degradedSince;
  return {
    ok: health.state !== "degraded",
    state: health.state,
    degradedForMs,
    restartAfterMs,
    lastHealthyAt: health.lastHealthyAt,
    lastIssueAt: health.lastIssueAt,
    lastIssue: health.lastIssue,
    issueCount: health.issueCount,
  };
}

export function classifyStreamLog(text) {
  if (!text.includes("[spectrum.stream]")) return null;
  if (text.includes("persistently failing")) return "degraded";
  if (text.includes("stream interrupted")) return "recovering";
  return null;
}

function stringifyConsoleArgs(args) {
  return args
    .map((arg) => (arg && arg.stack ? arg.stack : String(arg)))
    .join(" ");
}

export function installStreamLogClassifier(consoleObject, classify) {
  const originalError = consoleObject.error;
  const originalLog = consoleObject.log;

  consoleObject.error = (...args) => {
    classify(stringifyConsoleArgs(args));
    originalError.apply(consoleObject, args);
  };
  consoleObject.log = (...args) => {
    classify(stringifyConsoleArgs(args));
    originalLog.apply(consoleObject, args);
  };

  return () => {
    consoleObject.error = originalError;
    consoleObject.log = originalLog;
  };
}

export function inboundStreamErrorMessage(error) {
  const message = error && error.message ? error.message : String(error);
  let output = "photon-sidecar: inbound stream errored — restarting: " + message;

  // The Spectrum SDK surfaces Photon cloud CatchUpEvents failures as an
  // iMessage internal error. Local Hermes allowlists cannot cause or fix this:
  // inbound messages stop before they reach the gateway. Add an explicit hint
  // so operators know to retry/restart or escalate to Photon support instead
  // of chasing PHOTON_ALLOWED_USERS / pairing configuration.
  const details = String(error?.cause?.details || error?.details || "");
  const path = String(error?.cause?.path || error?.path || "");
  const code = String(error?.code || "");
  if (
    path.includes("EventService/CatchUpEvents") ||
    details.includes("Unknown server error occurred") ||
    (code === "internalError" && message.includes("Unknown server error"))
  ) {
    output +=
      " | Photon Spectrum CatchUpEvents returned an internal server error; " +
      "this is upstream of Hermes, so inbound iMessages may not be delivered " +
      "until Photon recovers or the stream is re-established.";
  }
  return output;
}
