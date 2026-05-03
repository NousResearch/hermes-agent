# Supported Chains and Common Tokens

## Chains Supported by Both Bankr and Symbiosis

| Chain | Chain ID | Native Token | Decimals |
|-------|----------|-------------|----------|
| Base | 8453 | ETH | 18 |
| Ethereum | 1 | ETH | 18 |
| Polygon | 137 | POL | 18 |
| Unichain | 130 | ETH | 18 |
| Solana | 5426 | SOL | 9 |

## Additional Chains (Symbiosis-only, not native to Bankr)

| Chain | Chain ID | Native Token | Decimals |
|-------|----------|-------------|----------|
| Arbitrum One | 42161 | ETH | 18 |
| Optimism | 10 | ETH | 18 |
| BNB Chain | 56 | BNB | 18 |
| Avalanche C-Chain | 43114 | AVAX | 18 |
| Gnosis | 100 | xDAI | 18 |
| zkSync Era | 324 | ETH | 18 |
| Linea | 59144 | ETH | 18 |
| Scroll | 534352 | ETH | 18 |
| Mantle | 5000 | MNT | 18 |
| Blast | 81457 | ETH | 18 |
| Mode | 34443 | ETH | 18 |
| Sei | 1329 | SEI | 18 |
| Gravity | 1625 | G | 18 |
| ZetaChain | 7000 | ZETA | 18 |
| Cronos | 25 | CRO | 18 |
| Bitcoin | 3652501241 | BTC | 8 |
| TON | 85918 | TON | 9 |
| Tron | 728126428 | TRX | 6 |

For the full list of 54+ chains, query: `GET https://api-v2.symbiosis.finance/crosschain/v1/chains`

## Common Token Addresses

### USDC

| Chain | Address | Decimals |
|-------|---------|----------|
| Base (8453) | 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913 | 6 |
| Ethereum (1) | 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48 | 6 |
| Polygon (137) | 0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359 | 6 |
| Arbitrum (42161) | 0xaf88d065e77c8cC2239327C5EDb3A432268e5831 | 6 |
| Optimism (10) | 0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85 | 6 |
| BNB Chain (56) | 0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d | 18 |
| Avalanche (43114) | 0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E | 6 |

### USDT

| Chain | Address | Decimals |
|-------|---------|----------|
| Ethereum (1) | 0xdAC17F958D2ee523a2206206994597C13D831ec7 | 6 |
| BNB Chain (56) | 0x55d398326f99059fF775485246999027B3197955 | 18 |
| Polygon (137) | 0xc2132D05D31c914a87C6611C10748AEb04B58e8F | 6 |
| Arbitrum (42161) | 0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9 | 6 |
| Optimism (10) | 0x94b008aA00579c1307B0EF2c499aD98a8ce58e58 | 6 |
| Avalanche (43114) | 0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7 | 6 |

### Native Gas Tokens

Use address `0x0000000000000000000000000000000000000000` with the appropriate decimals for the chain's native token (ETH=18, POL=18, BNB=18, SOL=9, BTC=8, etc.).

## Bitcoin (Special)

Bitcoin uses the Symbiosis v2 API (`/crosschain/v2/swap`) and returns a **deposit address** instead of calldata. The user must send BTC to the deposit address manually. The deposit address has an expiration time.

## Solana (Special)

Solana inbound swaps are limited to SOL and USDC as destination tokens. Outbound from Solana works for SOL and USDC as source.
