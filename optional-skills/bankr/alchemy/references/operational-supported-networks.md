---
id: references/operational-supported-networks.md
name: 'Supported Networks'
description: 'Alchemy supports multiple EVM and Solana networks. Always verify network availability in the dashboard for each product.'
tags:
  - alchemy
  - operational
  - operations
related:
  - node-json-rpc.md
  - solana-rpc.md
updated: 2026-03-02
---
# Supported Networks

## Summary
Alchemy supports multiple EVM and Solana networks. Always verify network availability in the dashboard for each product.

## Guidance
- Use chain-specific base URLs (e.g., `eth-mainnet`, `polygon-mainnet`).
- Use testnets for QA.
- Not all products are available on every chain.


## Spec-Derived Network Enums (Partial)
These lists are pulled directly from the repo's OpenAPI specs and may not be exhaustive. Always confirm in the dashboard.

### Notify API Networks (Uppercase)
```text
ETH_MAINNET
ETH_SEPOLIA
ETH_HOLESKY
ARB_MAINNET
ARB_SEPOLIA
ARBNOVA_MAINNET
MATIC_MAINNET
MATIC_MUMBAI
OPT_MAINNET
OPT_GOERLI
BASE_MAINNET
BASE_SEPOLIA
ZKSYNC_MAINNET
ZKSYNC_SEPOLIA
LINEA_MAINNET
LINEA_SEPOLIA
GNOSIS_MAINNET
GNOSIS_CHIADO
FANTOM_MAINNET
FANTOM_TESTNET
METIS_MAINNET
BLAST_MAINNET
BLAST_SEPOLIA
SHAPE_MAINNET
SHAPE_SEPOLIA
ZETACHAIN_MAINNET
ZETACHAIN_TESTNET
WORLDCHAIN_MAINNET
WORLDCHAIN_SEPOLIA
BNB_MAINNET
BNB_TESTNET
AVAX_MAINNET
AVAX_FUJI
SONEIUM_MAINNET
SONEIUM_MINATO
GEIST_POLTER
GEIST_MAINNET
STARKNET_MAINNET
STARKNET_SEPOLIA
STARKNET_GOERLI
INK_MAINNET
INK_SEPOLIA
ROOTSTOCK_MAINNET
ROOTSTOCK_TESTNET
SCROLL_MAINNET
SCROLL_SEPOLIA
MONAD_TESTNET
SONIC_MAINNET
SONIC_TESTNET
SETTLUS_SEPTESTNET
APECHAIN_MAINNET
APECHAIN_CURTIS
```

### Data API Networks (Lowercase)
```text
abstract-mainnet
abstract-testnet
adi-mainnet
adi-testnet
alchemy-internal
alchemy-sepolia
alchemyarb-fam
alchemyarb-sepolia
alterscope-mainnet
anime-mainnet
anime-sepolia
apechain-curtis
apechain-mainnet
aptos-mainnet
aptos-testnet
arb-mainnet
arb-sepolia
arbnova-mainnet
arc-testnet
astar-mainnet
avax-fuji
avax-mainnet
base-mainnet
base-sepolia
berachain-bepolia
berachain-mainnet
bitcoin-mainnet
bitcoin-signet
bitcoin-testnet
blast-mainnet
blast-sepolia
bnb-mainnet
bnb-testnet
bob-mainnet
bob-sepolia
boba-mainnet
boba-sepolia
botanix-mainnet
botanix-testnet
celestiabridge-mainnet
celestiabridge-mocha
celo-mainnet
celo-sepolia
citrea-mainnet
citrea-testnet
clankermon-mainnet
commons-mainnet
crossfi-mainnet
crossfi-testnet
degen-mainnet
degen-sepolia
earnm-mainnet
earnm-sepolia
edge-mainnet
edge-testnet
eth-holesky
eth-holeskybeacon
eth-hoodi
eth-hoodibeacon
eth-mainnet
eth-mainnetbeacon
eth-sepolia
eth-sepoliabeacon
flow-mainnet
flow-testnet
frax-hoodi
frax-mainnet
galactica-cassiopeia
galactica-mainnet
gensyn-mainnet
gensyn-testnet
gnosis-chiado
gnosis-mainnet
humanity-mainnet
humanity-testnet
hyperliquid-mainnet
hyperliquid-testnet
ink-mainnet
ink-sepolia
lens-mainnet
lens-sepolia
linea-mainnet
linea-sepolia
mantle-mainnet
mantle-sepolia
megaeth-mainnet
megaeth-testnet
metis-mainnet
mode-mainnet
mode-sepolia
monad-mainnet
monad-testnet
moonbeam-mainnet
mythos-mainnet
opbnb-mainnet
opbnb-testnet
openloot-sepolia
opt-mainnet
opt-sepolia
plasma-mainnet
plasma-testnet
polygon-amoy
polygon-mainnet
polygonzkevm-cardona
polygonzkevm-mainnet
polynomial-mainnet
polynomial-sepolia
race-mainnet
race-sepolia
risa-testnet
rise-testnet
ronin-mainnet
ronin-saigon
rootstock-mainnet
rootstock-testnet
scroll-mainnet
scroll-sepolia
sei-mainnet
sei-testnet
settlus-mainnet
settlus-septestnet
shape-mainnet
shape-sepolia
solana-devnet
solana-mainnet
soneium-mainnet
soneium-minato
sonic-blaze
sonic-mainnet
sonic-testnet
stable-mainnet
stable-testnet
standard-mainnet
starknet-mainnet
starknet-sepolia
story-aeneid
story-mainnet
sui-mainnet
sui-testnet
superseed-mainnet
superseed-sepolia
synd-mainnet
syndicate-manchego
tea-sepolia
tempo-testnet
tron-mainnet
tron-testnet
unichain-mainnet
unichain-sepolia
unite-mainnet
unite-testnet
worldchain-mainnet
worldchain-sepolia
worldl3-devnet
worldmobile-devnet
worldmobile-testnet
worldmobilechain-mainnet
xmtp-mainnet
xmtp-ropsten
xprotocol-mainnet
zetachain-mainnet
zetachain-testnet
zksync-mainnet
zksync-sepolia
zora-mainnet
zora-sepolia
```

## Related Files
- `node-json-rpc.md`
- `solana-rpc.md`

## Official Docs
- [Supported Chains](https://www.alchemy.com/docs/reference/node-supported-chains)
- [Chain APIs Overview](https://www.alchemy.com/docs/reference/chain-apis-overview)
