# Alchemy API Skill Map

Complete index of all reference files organized by product area. Use the Endpoint Selector in SKILL.md for quick routing to the most common tasks.

## Node
| File | Name | Short Description |
| --- | --- | --- |
| `references/node-overview.md` | node-apis | Core JSON-RPC and WebSocket APIs for EVM chains via Alchemy node endpoints, plus Debug/Trace and utility methods. Use when building EVM integrations that need standard RPC calls, real-time subscriptions, enhanced Alchemy methods, or execution-level tracing |
| `references/node-debug-api.md` | Debug API | Debug methods provide execution-level traces for a transaction or call. Use these for simulation, gas profiling, and internal call inspection |
| `references/node-enhanced-apis.md` | Enhanced APIs (Alchemy RPC Extensions) | Alchemy provides enhanced JSON-RPC methods (prefixed with `alchemy_`) that offer indexed, higher-level data without manual log scanning |
| `references/node-json-rpc.md` | JSON-RPC (EVM) | Use Alchemy's EVM JSON-RPC endpoints for standard blockchain reads and writes (e.g., `eth_call`, `eth_getLogs`, `eth_sendRawTransaction`). This is the baseline for any EVM integration |
| `references/node-trace-api.md` | Trace API | Trace APIs expose internal call data and state changes for transactions and blocks. Useful for analytics and auditing |
| `references/node-utility-api.md` | Utility API | Convenience RPC methods that reduce round trips for common tasks like bulk transaction receipt retrieval |
| `references/node-websocket-subscriptions.md` | WebSocket Subscriptions | Use WebSockets for real-time blockchain events without polling. Best for pending transactions, new blocks, and logs |

## Data
| File | Name | Short Description |
| --- | --- | --- |
| `references/data-overview.md` | data-apis | Higher-level Alchemy APIs for asset discovery, wallet analytics, transfer history, NFT data, and token pricing. Use when you need indexed blockchain data without raw RPC log scanning, including token balances, NFT ownership, portfolio views, price feeds, and transaction simulation |
| `references/data-nft-api.md` | NFT API | Query NFT ownership, metadata, collections, and contract-level info via Alchemy's NFT REST APIs |
| `references/data-portfolio-apis.md` | Portfolio APIs | Portfolio APIs provide consolidated wallet views (tokens, NFTs, and transaction history) across multiple networks in single requests |
| `references/data-prices-api.md` | Prices API | Query token prices for current and historical data using Alchemy's Prices API |
| `references/data-simulation-api.md` | Simulation API | Simulate transactions before submitting them on-chain. Use this for safety checks and user previews |
| `references/data-token-api.md` | Token API | Fetch token balances, metadata, and allowances without manual contract calls. Token API methods are exposed as Alchemy JSON-RPC methods |
| `references/data-transfers-api.md` | Transfers API | Query historical transfers across ERC-20/721/1155 and native transfers without manual log scanning |

## Webhooks
| File | Name | Short Description |
| --- | --- | --- |
| `references/webhooks-overview.md` | webhooks | Push-based delivery of blockchain events via Alchemy Notify API. Use when you need real-time notifications for address activity, NFT transfers, or custom on-chain events instead of polling. Covers webhook creation, payload formats, signature verification, and filtering |
| `references/webhooks-address-activity.md` | Address Activity Webhooks | Address activity webhooks notify you when specified addresses send or receive assets |
| `references/webhooks-custom-webhooks.md` | Custom Webhooks (GraphQL) | Custom webhooks allow flexible event filtering using a GraphQL query hosted by Alchemy |
| `references/webhooks-details.md` | Webhooks Overview (Notify) | Notify webhooks push blockchain events to your server so you don't need to poll. They are ideal for near real-time pipelines |
| `references/webhooks-nft-activity.md` | NFT Activity Webhooks | Receive notifications for NFT transfers, mints, and burns for specified contracts or collections |
| `references/webhooks-verify-signatures.md` | Verify Webhook Signatures | Always verify webhook signatures to ensure payloads are authentic and untampered |
| `references/webhooks-webhook-payloads.md` | Webhook Payloads | Webhook payloads include event metadata plus event-specific fields. Treat them as untrusted input and validate carefully |
| `references/webhooks-webhook-types.md` | Webhook Types | Webhook types determine what events you receive and how they're filtered |

## Solana
| File | Name | Short Description |
| --- | --- | --- |
| `references/solana-overview.md` | solana | Solana-specific APIs including standard JSON-RPC, Digital Asset Standard (DAS) for NFTs and compressed assets, and wallet integration. Use when building Solana applications that need RPC access, NFT/asset queries, or Solana wallet tooling. For high-throughput streaming, see the yellowstone-grpc skill |
| `references/solana-das-api.md` | Solana DAS (Digital Asset Standard) API | DAS provides normalized access to Solana NFT and compressed asset data |
| `references/solana-grpc-best-practices.md` | Yellowstone Best Practices | Practical guidance to keep Yellowstone consumers reliable and efficient |
| `references/solana-grpc-details.md` | Yellowstone gRPC Overview | Yellowstone gRPC provides high-throughput Solana data streams for blocks, transactions, accounts, and slots. Use this for real-time indexing at scale |
| `references/solana-grpc-examples.md` | Yellowstone Examples | Minimal examples for connecting and subscribing. The exact client depends on your gRPC stack |
| `references/solana-grpc-overview.md` | yellowstone-grpc | High-throughput Solana event streaming via Yellowstone gRPC. Use when you need near real-time block, transaction, account, or slot streams at scale. Covers subscription configuration, filtering, backpressure handling, and reconnection strategies |
| `references/solana-grpc-subscribe-accounts.md` | Subscribe Accounts | Account streams deliver updates when account state changes |
| `references/solana-grpc-subscribe-blocks.md` | Subscribe Blocks | Block streams provide full block data with transactions and metadata |
| `references/solana-grpc-subscribe-request.md` | Subscribe Request | The subscribe request configures what streams and filters you want from Yellowstone |
| `references/solana-grpc-subscribe-slots.md` | Subscribe Slots | Slot updates provide heartbeat-style updates and can be used to track chain progress |
| `references/solana-grpc-subscribe-transactions.md` | Subscribe Transactions | Transaction streams deliver raw or decoded transaction data in near real-time |
| `references/solana-rpc.md` | Solana JSON-RPC | Standard Solana JSON-RPC endpoints for account, program, and transaction data |
| `references/solana-wallets.md` | Solana Wallet Integration | High-level guidance for integrating Solana wallets and signing transactions |

## Wallets
| File | Name | Short Description |
| --- | --- | --- |
| `references/wallets-overview.md` | wallets | Integration guide for Alchemy Wallets and smart wallet tooling including Account Kit, account abstraction, bundler, gas manager, and paymaster. Use when building wallet onboarding flows, sponsoring gas, or integrating smart wallets into your application |
| `references/wallets-account-kit.md` | Account Kit | Account Kit is Alchemy's wallet SDK for onboarding users and managing wallet UX. Use it for embedded wallet flows or seamless authentication |
| `references/wallets-bundler.md` | Bundler | A bundler aggregates and submits account abstraction user operations. Use this when integrating smart wallets |
| `references/wallets-details.md` | Wallets Overview | Alchemy Wallets tooling helps developers embed or integrate wallets with minimal infrastructure. This section is intentionally basic and focuses on integration touchpoints |
| `references/wallets-gas-manager.md` | Gas Manager | Gas Manager (paymaster) enables gas sponsorship and cost control for smart wallet flows |
| `references/wallets-smart-wallets.md` | Smart Wallets | Smart wallets (account abstraction) enable programmable accounts with features like session keys, batched transactions, and gas sponsorship |
| `references/wallets-solana-notes.md` | Solana Wallet Notes | Solana wallet integration differs from EVM. Use Solana-specific tooling and RPC semantics |
| `references/wallets-supported-chains.md` | Wallet Supported Chains | Wallet tooling may support a subset of chains compared to raw RPC. Always confirm chain support before launch |
| `references/wallets-wallet-apis.md` | Wallet APIs | High-level wallet APIs enable programmatic wallet operations such as signing, transaction preparation, or account management. This guide stays minimal and focuses on integration awareness |

## Rollups
| File | Name | Short Description |
| --- | --- | --- |
| `references/rollups-overview.md` | rollups | High-level overview of Alchemy Rollups for deploying custom L2/L3 chains. Use when exploring rollup deployment options or understanding what Alchemy Rollups offers. This section is intentionally minimal as rollup deployment is not a typical API integration task |
| `references/rollups-details.md` | Rollups Overview | Alchemy Rollups provide infrastructure for teams building their own rollup chains. This section is intentionally high-level because rollup deployment typically requires a sales or solutions process rather than a purely programmatic integration |

## Recipes
| File | Name | Short Description |
| --- | --- | --- |
| `references/recipes-overview.md` | recipes | End-to-end runnable workflows that combine multiple Alchemy APIs to achieve real product goals. Includes token balances, NFT queries, transfer history, portfolio views, price feeds, transaction simulation, pending tx monitoring, and webhook setup. Use when you need a complete integration pattern, not just a single API call |
| `references/recipes-get-nft-metadata.md` | Recipe: Get NFT Metadata | Fetch NFT metadata for a token or collection using Alchemy NFT endpoints |
| `references/recipes-get-nft-ownership.md` | Recipe: Get NFT Ownership | Retrieve wallet-level NFT ownership with pagination and filtering |
| `references/recipes-get-portfolio.md` | Recipe: Get Full Portfolio | Build a consolidated wallet portfolio view across supported networks |
| `references/recipes-get-prices-current-historical.md` | Recipe: Get Current and Historical Prices | Query spot and historical token prices for analytics and UI displays |
| `references/recipes-get-token-balances.md` | Recipe: Get Token Balances | Get ERC-20 token balances for a wallet and handle paginated responses |
| `references/recipes-get-token-metadata.md` | Recipe: Get Token Metadata | Resolve token metadata (symbol, decimals, logo) for contract addresses |
| `references/recipes-get-transfers-history.md` | Recipe: Get Transfer History | Pull historical transfer activity with category and block-range filters |
| `references/recipes-simulate-transaction.md` | Recipe: Simulate a Transaction | Simulate transaction effects before submission for risk checks and UX previews |
| `references/recipes-subscribe-pending-txs.md` | Recipe: Subscribe to Pending Transactions | Stream pending transactions in real time via WebSocket subscriptions |
| `references/recipes-webhook-address-activity.md` | Recipe: Address Activity Webhook | Configure address activity webhooks and verify delivered signatures |

## Operational
| File | Name | Short Description |
| --- | --- | --- |
| `references/operational-overview.md` | operational | Operational guidance for securely and reliably integrating Alchemy APIs. Covers API key management, JWT authentication, rate limits, compute unit budgeting, monitoring, alerting, and production readiness. Use before deploying any Alchemy integration to production |
| `references/operational-alerts.md` | Alerts | Set alerts to catch rate limit issues, spikes in usage, or webhook failures |
| `references/operational-allowlists.md` | Allowlists | Allowlisting restricts API key usage to approved IPs or domains, reducing key abuse risk |
| `references/operational-auth-and-keys.md` | Auth and API Keys | Alchemy uses API keys for most products. Keep keys server-side and scope them to environments |
| `references/operational-best-practices.md` | Production Best Practices | Operational checklist for deploying Alchemy-backed services |
| `references/operational-dashboard-tools.md` | Dashboard Tools | Use the Alchemy dashboard to create apps, manage keys, track usage, and configure webhooks |
| `references/operational-jwt-and-header-auth.md` | JWT and Header-Based Auth | Some Alchemy APIs use header-based auth (e.g., JWTs or API tokens) instead of API keys in URLs. Use this for server-side integrations where you need stronger access control |
| `references/operational-pricing-and-plans.md` | Pricing and Plans | Alchemy plans are primarily based on compute units and product access. Choose a plan that matches your expected traffic and data needs |
| `references/operational-rate-limits-and-compute-units.md` | Rate Limits and Compute Units | Alchemy meters usage using compute units (CU) and enforces per-key rate limits. Plan your request patterns accordingly |
| `references/operational-request-logs.md` | Request Logs | Request logs help debug failed calls and monitor latency |
| `references/operational-roles-and-sso.md` | Roles and SSO | Team access control ensures API keys and billing settings are managed safely |
| `references/operational-sandbox.md` | Sandbox / Test Environment | Use testnets or a separate Alchemy app for safe development and QA |
| `references/operational-supported-networks.md` | Supported Networks | Alchemy supports multiple EVM and Solana networks. Always verify network availability in the dashboard for each product |

## Ecosystem
| File | Name | Short Description |
| --- | --- | --- |
| `references/ecosystem-overview.md` | ecosystem | Popular open-source libraries that pair well with Alchemy for building crypto applications. Covers EVM tools (viem, ethers, web3.js, wagmi, RainbowKit, Hardhat, Foundry) and Solana tools (solana/web3.js, spl-token, Anchor, Metaplex). Use when choosing or configuring a client library for your Alchemy integration |
| `references/ecosystem-anchor.md` | Anchor | Anchor is a Solana framework for building programs with a higher-level Rust and TypeScript SDK |
| `references/ecosystem-ethers.md` | Ethers.js | Ethers.js is a widely used Ethereum library for wallets, providers, and contract interactions |
| `references/ecosystem-foundry.md` | Foundry | Foundry is a fast, Rust-based Ethereum toolchain for testing and deployment |
| `references/ecosystem-hardhat.md` | Hardhat | Hardhat is a popular Ethereum development environment for compiling, testing, and deploying contracts |
| `references/ecosystem-metaplex.md` | Metaplex | Metaplex provides NFT standards and tooling on Solana, including metadata and token standards |
| `references/ecosystem-rainbowkit.md` | RainbowKit | RainbowKit is a wallet connection UI kit commonly used with wagmi |
| `references/ecosystem-solana-web3js.md` | @solana/web3.js | The primary JavaScript library for Solana RPC, transactions, and accounts |
| `references/ecosystem-spl-token.md` | SPL Token | SPL Token is the standard token program on Solana and includes tools for minting and token account management |
| `references/ecosystem-viem.md` | Viem | Viem is a modern TypeScript Ethereum client library with strong types and a functional API |
| `references/ecosystem-wagmi.md` | wagmi | wagmi provides React hooks for Ethereum, commonly used with wallet connectors and UI kits |
| `references/ecosystem-web3js.md` | Web3.js | Web3.js is a long-standing Ethereum JavaScript library for providers and contracts |
