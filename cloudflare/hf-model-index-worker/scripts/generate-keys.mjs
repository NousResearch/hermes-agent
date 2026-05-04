import { writeFileSync } from "node:fs";
import { webcrypto } from "node:crypto";

const keyPair = await webcrypto.subtle.generateKey(
  {
    name: "ECDSA",
    namedCurve: "P-256"
  },
  true,
  ["sign", "verify"]
);

const privateJwk = await webcrypto.subtle.exportKey("jwk", keyPair.privateKey);
const publicJwk = await webcrypto.subtle.exportKey("jwk", keyPair.publicKey);

writeFileSync(".private-jwk.json", JSON.stringify(privateJwk));
writeFileSync(".public-jwk.json", JSON.stringify(publicJwk, null, 2));

console.log("Wrote .private-jwk.json and .public-jwk.json");
console.log("Put .private-jwk.json into Cloudflare as PRIVATE_JWK.");
console.log("Embed .public-jwk.json in the Android app.");
