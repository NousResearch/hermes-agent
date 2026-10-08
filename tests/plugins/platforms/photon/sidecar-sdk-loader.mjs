// Recording-only SDK boundary; the real sidecar server and handlers run unchanged.
const sdk = `
process.on("message", (message) => {
  if (message.type === "barrier") process.send?.(message);
});
export class NotFoundError extends Error {
  constructor(message, options) {super(message); Object.assign(this, options); this.name = "NotFoundError";}
}
export class ValidationError extends Error {
  constructor(message, options) {super(message); Object.assign(this, options); this.name = "ValidationError";}
}
export const ErrorCode = {messageNotFound: "messageNotFound"};
const send = async (publicBuilder) => {
  const builder = await publicBuilder.build();
  process.send?.({type: "send", builder, publicBuilderKeys: Object.keys(publicBuilder)});
  const inner = builder.type === "reply" ? builder.content : builder;
  const failure = inner.text ?? inner.path;
  if (failure?.startsWith("voice-refusal")) {
    const message = "[upstream] is_audio_message with reply_to is not supported by the IMAgentKit send path";
    const options = {grpcCode: 9, code: "internalError", retryable: false};
    if (failure === "voice-refusal-grpc") options.grpcCode = 3;
    if (failure === "voice-refusal-code") options.code = "invalidArgument";
    if (failure === "voice-refusal-retryable") options.retryable = true;
    if (failure === "voice-refusal-missing-retryable") delete options.retryable;
    const e = failure === "voice-refusal-spoof" ? new Error(message) :
      new ValidationError(failure === "voice-refusal-ambiguous" ? message + "; delivery unknown" : message, options);
    if (failure === "voice-refusal-spoof") Object.assign(e, options, {name: "ValidationError"});
    if (builder.type === "reply" || failure !== "voice-refusal") throw e;
  }
  if (inner.text === "spoof" || inner.path === "spoof") {
    const e = new Error("missing"); Object.assign(e, {name: "NotFoundError", code: "messageNotFound", grpcCode: 5, retryable: false}); throw e;
  }
  if (inner.text === "wrong-code" || inner.path === "wrong-code") {
    throw new NotFoundError("missing chat", {code: "chatNotFound", grpcCode: 5, retryable: false});
  }
  if (inner.text === "timeout" || inner.path === "timeout") throw new Error("network timeout");
  if (inner.text === "drop" || inner.path === "drop") throw new Error("socket hang up");
  if (inner.text === "503" || inner.path === "503") throw new Error("upstream 503");
  if (builder.type === "reply" && builder.target.id === "unsupported") return undefined;
  if (builder.type === "reply" && ((builder.target.id === "stale" && inner.text !== "cache-preserved") || inner.text === "reject-caption")) {
    throw new NotFoundError("message missing", {code: "messageNotFound", grpcCode: 5, retryable: false});
  }
  process.send?.({type: "delivered", builder});
  return {id: "sent"};
};
const space = {id: "space", send, getMessage: async (id) => {
  process.send?.({type: "lookup", id});
  if (id === "missing") return undefined;
  return {id, content: {type: "text", text: "anchor"}, direction: "inbound"};
}};
export async function Spectrum() {
  return {stop: async () => {}, messages: {[Symbol.asyncIterator]: async function* () {
    yield [space, {id: "stale", space, sender: {id: "sender"}, content: {type: "text", text: "cached"}, direction: "inbound", timestamp: new Date()}];
    await new Promise(() => {});
  }}};
}
// spectrum-ts 12.7.0 exposes build-only builders, not resolved content.
export const text = (text) => ({build: async () => ({type: "text", text})});
export const markdown = (text) => ({build: async () => ({type: "markdown", text})});
export const attachment = (path, opts) => ({build: async () => ({type: "attachment", path, ...opts})});
export const voice = (path, opts) => ({build: async () => ({type: "voice", path, ...opts})});
export const reply = (content, target) => ({build: async () => ({type: "reply", content: await content.build(), target})});
export const richlink = text;
export const typing = text;
export const poll = text;
export function imessage() {return {space: {get: async () => space, create: async () => space}};}
imessage.config = () => ({});
export const effect = text;
`;
export async function resolve(specifier, context, nextResolve) {
  if (specifier === "spectrum-ts" || specifier === "spectrum-ts/providers/imessage" || specifier === "@photon-ai/advanced-imessage/grpc") {
    return {url: "data:text/javascript," + encodeURIComponent(sdk), shortCircuit: true};
  }
  return nextResolve(specifier, context);
}
