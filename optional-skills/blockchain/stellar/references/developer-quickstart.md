# Stellar developer quickstart

Companion to the `stellar` skill's read-only client. Read this when the user wants
to *build* on Stellar: deploy a contract, call it from code, or move assets. It is a
condensed map, not the full guide. The maintained, in-depth skill set is
[stellar/stellar-dev-skill](https://github.com/stellar/stellar-dev-skill) (also at
[skills.stellar.org](https://skills.stellar.org)), one skill per area:
`smart-contracts`, `dapp`, `data`, `assets`, `standards`, `agentic-payments`,
`cross-chain`, `zk-proofs`. Install the one you need next to this skill:

```bash
hermes skills install stellar/stellar-dev-skill/skills/smart-contracts
hermes skills install stellar/stellar-dev-skill/skills/dapp
```

For live lookups (current docs, protocol version, ecosystem data) the Stellar
community runs Raven, a remote MCP server. Add it under `mcp_servers` in
`~/.hermes/config.yaml`:

```yaml
mcp_servers:
  stellar-raven:
    url: https://raven.stellar.buzz/mcp
```

## Mental model

- **Two layers.** Classic Stellar: accounts (`G...`), assets issued by accounts
  (`CODE:ISSUER`), trustlines, payments, the built-in DEX. Soroban: WASM smart
  contracts (`C...`) written in Rust with `soroban-sdk`. A Stellar Asset Contract
  (SAC) exposes any classic asset to contracts as a SEP-41 token, so USDC has both a
  classic issuer `G...` and a SAC `C...`.
- **Accounts must be funded to exist.** The first XLM deposit creates the account.
  Reserves: 1 XLM base plus 0.5 XLM per trustline, offer, signer, or data entry.
  Testnet accounts get free XLM from friendbot (`stellar_client.py fund`).
- **Two APIs.** Stellar RPC (`getLedgerEntries`, `simulateTransaction`,
  `sendTransaction`, `getEvents`; about 7 days of history) is for contracts and
  all new work. Horizon (REST, full history, streaming) is for classic data.
- **Storage is rented.** Every contract entry has a TTL in ledgers (about 5 s each).
  Expired persistent entries are archived and must be restored; expired temporary
  entries are gone. Check with `stellar_client.py contract <C...>`.
- **Networks.** Mainnet passphrase `Public Global Stellar Network ; September 2015`,
  testnet `Test SDF Network ; September 2015`. Testnet resets periodically.
- **Versions.** The `soroban-sdk` major version tracks the live protocol version
  (`stellar_client.py stats` prints it). Pin the SDK major that matches your target
  network; check [crates.io/crates/soroban-sdk](https://crates.io/crates/soroban-sdk)
  for the current release rather than trusting any pinned example.

## Toolchain

```bash
# Stellar CLI (keys, contracts, networks). Homebrew or cargo.
brew install stellar-cli
cargo install --locked stellar-cli

# Rust target for contracts (Rust 1.84+)
rustup target add wasm32v1-none

# JS SDK needs Node 22+
npm install @stellar/stellar-sdk
```

## Testnet identity

```bash
stellar keys generate alice --network testnet --fund   # creates + funds via friendbot
stellar keys address alice                              # prints the G... address
```

Secret keys stay in the CLI keystore (or the user's wallet). Never paste an `S...`
seed into the conversation; the `stellar` skill refuses them as input.

## Contract loop (Rust)

```bash
stellar contract init my-contract && cd my-contract
stellar contract build                                   # -> target/wasm32v1-none/release/*.wasm

stellar contract deploy \
  --wasm target/wasm32v1-none/release/my_contract.wasm \
  --source-account alice --network testnet \
  -- --admin alice                                       # constructor args after `--`

stellar contract invoke --id CONTRACT_ID --source-account alice --network testnet -- increment
stellar contract info interface --id CONTRACT_ID --network testnet   # read the ABI
```

Contract essentials: `#![no_std]`, state in `env.storage().instance()/persistent()/
temporary()`, `address.require_auth()` for authorization, `#[contracterror]` enums
for typed errors, `#[contractevent]` for events, `__constructor` for one-time init,
`extend_ttl` so state is not archived. Tests run in-process with
`Env::default()` and `env.mock_all_auths()`. Full patterns, testing layers, and the
pre-mainnet security checklist: `smart-contracts` in stellar-dev-skill.

After deploying, verify with the read-only client:

```bash
python ~/.hermes/skills/blockchain/stellar/scripts/stellar_client.py contract CONTRACT_ID --network testnet
```

## Call a contract from JavaScript

```typescript
import { contract, Keypair, Networks } from "@stellar/stellar-sdk";

const keypair = Keypair.fromSecret(process.env.STELLAR_SECRET!);   // never log it
const client = await contract.Client.from({
  contractId: "C...",
  rpcUrl: "https://soroban-testnet.stellar.org",
  networkPassphrase: Networks.TESTNET,
  publicKey: keypair.publicKey(),
  ...contract.basicNodeSigner(keypair, Networks.TESTNET),           // browser: wallet signer
});

const tx = await (client as any).increment();   // simulated: tx.result is the predicted return
const { result } = await tx.signAndSend();      // signs, submits, polls to completion
```

`contract.Client` reads the interface from the network, converts arguments to and
from XDR, and simulates before signing. In a browser, pass the wallet's
`signTransaction` (Freighter, Stellar Wallets Kit) instead of a keypair. Details:
`dapp` in stellar-dev-skill.

## Classic payment from Node

```typescript
import * as S from "@stellar/stellar-sdk";

const horizon = new S.Horizon.Server("https://horizon-testnet.stellar.org");
const source = S.Keypair.fromSecret(process.env.STELLAR_SECRET!);
const account = await horizon.loadAccount(source.publicKey());
const tx = new S.TransactionBuilder(account, { fee: S.BASE_FEE, networkPassphrase: S.Networks.TESTNET })
  .addOperation(S.Operation.payment({ destination: "G...", asset: S.Asset.native(), amount: "10" }))
  .setTimeout(60)
  .build();
tx.sign(source);
const { hash } = await horizon.submitTransaction(tx);
```

For a non-XLM asset the destination needs a trustline first
(`Operation.changeTrust({ asset: new Asset("USDC", issuer) })`), or the payment
fails with `op_no_trust`. Check the result with `stellar_client.py tx <hash>`.

## USDC addresses

| Network | Classic issuer (trustlines, payments) | SAC (contract calls) |
|---|---|---|
| mainnet | `GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN` | `CCW67TSZV3SSS2HXMBQ5JFGCKJNXKZM7UQUWUZPUTHXSTZLEO7SJMI75` |
| testnet | `GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5` | `CBIELTK6YBZJU5UP2WWQEUCYKLPU6AUNZ2BQ4WWFEIE3USCIHMXQDAMA` |

Testnet USDC comes from the [Circle faucet](https://faucet.circle.com/) (web only).

## Which SEP?

| Need | Standard |
|---|---|
| Publish issuer/anchor metadata at `/.well-known/stellar.toml` | SEP-1 |
| `name*domain` addresses | SEP-2 |
| Deposit/withdraw API, hosted deposit/withdraw UI | SEP-6, SEP-24 |
| Payment request / signing links (`web+stellar:`) | SEP-7 |
| Prove control of an account to a server (challenge tx) | SEP-10 |
| KYC data exchange | SEP-12 |
| Cross-border payments between anchors | SEP-31 |
| Token interface every Soroban token (and SAC) implements | SEP-41 |
| Agent-to-API payments over HTTP 402 (x402, MPP) on Stellar | `agentic-payments` in stellar-dev-skill |

Status and text live in [stellar/stellar-protocol](https://github.com/stellar/stellar-protocol);
check there before relying on a draft.

## References

- Docs: https://developers.stellar.org (RPC methods under Data, contracts under Build)
- Stellar Lab (build, sign, inspect XDR, network limits): https://lab.stellar.org
- Explorer: https://stellar.expert
- Mainnet RPC providers: https://developers.stellar.org/docs/data/apis/rpc/providers
- Example contracts: https://github.com/stellar/soroban-examples
- JS SDK reference: https://stellar.github.io/js-stellar-sdk/
