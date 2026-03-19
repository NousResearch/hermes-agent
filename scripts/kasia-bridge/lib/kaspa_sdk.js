import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const bridgeDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const kaspaEntry = resolve(bridgeDir, "vendor", "kaspa-wasm", "kaspa.js");

if (!existsSync(kaspaEntry)) {
  throw new Error(
    "Pinned kaspa-wasm runtime is missing. Run `npm install` in scripts/kasia-bridge."
  );
}

const kaspa = require(kaspaEntry);

export default kaspa;
export const {
  Address,
  addressFromScriptPublicKey,
  ConnectStrategy,
  Encoding,
  Generator,
  Mnemonic,
  NetworkId,
  PaymentOutput,
  PrivateKeyGenerator,
  RpcClient,
  UtxoContext,
  UtxoEntries,
  UtxoProcessor,
  XPrv,
  XOnlyPublicKey,
  kaspaToSompi,
} = kaspa;
