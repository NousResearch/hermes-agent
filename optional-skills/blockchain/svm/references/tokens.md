# Token Extensions & DeFi Primitives

Solana's token ecosystem has evolved beyond the original SPL Token program. Token-2022 adds programmable extensions (transfer fees, confidential transfers, metadata) directly into the token standard. The liquid staking ecosystem turns staked SOL into productive collateral. Stablecoins and RWAs represent Solana's integration with traditional finance — an area where Solana's speed and low costs give it a structural advantage.

## Key Concepts

- **SPL Token** — the original Solana token standard; simple, battle-tested, supports mint + transfer + approve; most existing tokens (USDC, BONK, etc.) use this
- **Token-2022** — the new token program with extension architecture; each mint can opt into specific extensions at creation time; not backwards-compatible with SPL Token
- **Transfer fees** — Token-2022 extension; charges a configurable basis-point fee on each transfer; fees accumulate in the recipient's token account and can be harvested by the fee authority
- **Confidential transfers** — Token-2022 extension using ElGamal encryption and ZK proofs; balances and transfer amounts are hidden from on-chain observers; useful for compliant privacy
- **Metadata pointer** — Token-2022 extension; stores a metadata address (e.g., Metaplex) or inline metadata in the mint account
- **Permanent delegate** — Token-2022 extension; designates an address that can transfer or burn tokens from any holder's account; used for regulated assets and compliance
- **LSTs (Liquid Staking Tokens)** — SPL tokens representing staked SOL; accrue staking rewards while remaining tradeable; examples: jitoSOL (Jito), mSOL (Marinade), bSOL (BlazeStake), hSOL (Helius)
- **Stablecoin landscape** — USDC (Circle, native SPL), USDT (Tether, bridged), PYUSD (PayPal, native), USDe (Ethena, yield-bearing synthetic); Solana has the most diverse native stablecoin ecosystem of any non-Ethereum chain
- **RWAs (Real World Assets)** — tokenized bonds, equities, commodities, or real estate on Solana; typically use Token-2022 with transfer restrictions and permanent delegate for compliance

## Blog Posts

Use `fetchHeliusBlog` with these slugs:

- `what-is-token-2022` — Token-2022 deep dive: all extensions explained, migration considerations from SPL Token, and when to use each extension; **note**: confidential transfers are covered at survey level only — no dedicated deep-dive post exists for the ZK cryptography or pending/available balance model
- `lsts-on-solana` — LST ecosystem: how liquid staking works, the major protocols (Jito, Marinade, BlazeStake), APY mechanics, and the risks
- `solanas-stablecoin-landscape` — Stablecoin overview: all major stablecoins on Solana, their mechanisms (fiat-backed, algorithmic, synthetic), and market dynamics
- `solana-real-world-assets` — RWAs on Solana: tokenization approaches, compliance tooling, and the projects building in this space

## Relevant SIMDs

Use `getSIMD` for recent Token-2022 extension proposals and any changes to the token program interface.

## Source Code Entry Points

**Token-2022 source is not in agave.** `readSolanaSourceFile` is scoped to `anza-xyz/agave` and Firedancer only. Token-2022 and all SPL token programs live in [`solana-program/token-2022`](https://github.com/solana-program/token-2022) — a separate repo the tool cannot reach. Skip source code fetches for this topic and rely on `fetchHeliusBlog` instead.

## Solana Docs

Try `searchSolanaDocs` with: "token 2022 extensions", "transfer fees", "confidential transfers", "liquid staking", "token program"

## See Also

- `references/accounts.md` — token accounts are Solana accounts; understanding ownership and rent applies directly
- `references/development.md` — building programs that interact with Token-2022
