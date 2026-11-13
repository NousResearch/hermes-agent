// Run only in an isolated test child: no project lookup or external sockets.
import http from "node:http";
import { mock } from "node:test";
import { createLogger } from "@photon-ai/otel";
import { ChannelCredentials } from "nice-grpc";
import { ClientError, Status } from "nice-grpc-common";

// Exercise the installed SDK's request/error mapping. The service fixture
// models the proxy's message-resource contract, not a client-side validator:
// the SDK forwards the opaque messageGuid unchanged to GetMessage.
const factory = {
  use() { return this; },
  create() {
    return {
      async getMessage({ messageGuid }) {
        if (process.env.TEST_PROBE_FAILURE === "unavailable") {
          throw new ClientError("GetMessage", Status.UNAVAILABLE, "unavailable");
        }
        if (!/^spc-msg-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(messageGuid)) {
          throw new ClientError("GetMessage", Status.INVALID_ARGUMENT, "Expected message resource GUID");
        }
        throw new ClientError("GetMessage", Status.NOT_FOUND, "message not found");
      },
    };
  },
};
mock.module("nice-grpc", { namedExports: {
  ChannelCredentials,
  createChannel: () => ({ close() {} }),
  createClientFactory: () => factory,
} });
const { createGrpcClient } = await import("@photon-ai/advanced-imessage/grpc");
const client = createGrpcClient({ address: "test.invalid", tls: false });
const provider = () => ({ space: {
  get: async () => ({ getMessage: id => client.messages.get(id) }),
} });
provider.config = config => config;
mock.module("spectrum-ts/providers/imessage", {
  namedExports: { imessage: provider },
});
mock.module("spectrum-ts", { namedExports: {
  Spectrum: async () => ({
    messages: { [Symbol.asyncIterator]() {
      return { next: () => new Promise(() => {}) };
    } },
    stop: () => client.close(),
  }),
  attachment: value => value,
  voice: value => value,
  text: value => value,
  markdown: value => value,
  richlink: value => value,
  typing: value => value,
  poll: value => value,
} });
mock.module(new URL("../../patch-spectrum-mixed-attachments.mjs", import.meta.url), {
  namedExports: { patchSpectrumTs: () => ({ patched: false }) },
});

const createServer = http.createServer;
mock.method(http, "createServer", (...args) => {
  const server = createServer(...args);
  server.once("listening", () => process.send({ port: server.address().port }));
  return server;
});
const log = createLogger("spectrum.stream");
process.on("message", ({ warnings }) => {
  for (let i = 0; i < warnings; i++) {
    log.warn("stream interrupted; reconnecting");
  }
  process.send({ warnings });
});
await import("../../index.mjs");
