// Recording-only SDK boundary; the real sidecar server and handlers run unchanged.
const sdk = `
process.on("message", (message) => {
  if (message.type === "barrier") process.send?.(message);
});
export class NotFoundError extends Error {
  constructor(message, options) {super(message); Object.assign(this, options); this.name = "NotFoundError";}
}
export const ErrorCode = {messageNotFound: "messageNotFound"};
const send = async (builder) => {
  process.send?.({type: "send", builder});
  const inner = builder.type === "reply" ? builder.content : builder;
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
  if (builder.type === "reply" && (builder.target.id === "stale" || inner.text === "reject-caption")) {
    throw new NotFoundError("message missing", {code: "messageNotFound", grpcCode: 5, retryable: false});
  }
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
export const text = (text) => ({type: "text", text});
export const markdown = (text) => ({type: "markdown", text});
export const attachment = (path, opts) => ({type: "attachment", path, ...opts});
export const voice = (path, opts) => ({type: "voice", path, ...opts});
export const reply = (content, target) => ({type: "reply", content, target});
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
