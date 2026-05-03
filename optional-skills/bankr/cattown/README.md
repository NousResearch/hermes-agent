# Cat Town — Bankr Skill

Interact with **Cat Town**, a Farcaster-native game world on Base, from Bankr.

## What it does

Gives an agent everything it needs to:

- **Stake KIBBLE** into the RevenueShare contract and earn a share of weekly fishing + gacha revenue.
- **Read a user's position** — staked amount, pending rewards, pool share, unlock state, time left until withdraw.
- **Exit correctly** — `unlock()` → 14-day wait → `unstake()`, with the right user-facing messaging (share drop to 0%, opportunity cost, `relock()` escape hatch).
- **Read Cat Town's live world state** — season, time of day, weather, weekend flag — via the GameData contract.
- **Answer "what can I catch in this weather?"** — combine world state with the public item-truth catalog to surface weather / season / time-of-day exclusive fishing drops.
- **Read the boutique's daily 3-item rotation** with KIBBLE prices converted to USD via the Kibble Price Oracle.
- **Report live fishing competition state** (weekly Sat–Mon): running time, weather, participants, 10/80/10 prize-pool split with top-10 payouts, with a reminder offer + "tell me about the last one" fallback when off-season.
- **Answer KIBBLE tokenomics** — % burned (of total), % staked (of circulating), and live APY — mirroring the game's Jasper NPC math.
- **Claim the weekly fish-raffle free ticket** (Paulie's draw Fri 20:00 UTC), report the live prize pool and tier, and compute the user's chance to win from the current leaderboard.
- **Submit gacha pulls** with the async VRF pattern (pay tx → poll `/v2/items/capsule` for new token ids), handle the 100-per-day cap, and quote the USD-denominated KIBBLE cost per pull.
- **Value and sell items** — look up an item's `sellValue` (US cents), convert to KIBBLE via the oracle, and batch-sell Treasures + Collectibles through the vendor (V2-minter items only, 5% merchant fee).
- **Query the staking leaderboard and weekly revenue-deposit history** via public unauthenticated endpoints on `api.cat.town`.

## Install

```
install the cattown skill from https://github.com/cattownbase/cattown-bankr-skills/tree/main/cattown
```

## Layout

```
cattown/
├── SKILL.md              entry — triggers, flows, routing
├── README.md             this file
├── docs.md               comprehensive protocol overview (AI-readable)
└── references/
    ├── staking/          RevenueShare contract + api.cat.town staking endpoints
    ├── world/            GameData contract + weekly calendar
    ├── fishing/          fishing drops + weekly competition leaderboard
    ├── fish-raffle/      Paulie's weekly raffle contracts + tickets API
    ├── boutique/         daily boutique rotation + KIBBLE price oracle
    ├── gacha/            gacha machine + async VRF capsule polling
    ├── sell-items/       vendor sell flow (V2 minter items, 5% tax)
    └── kibble/           KIBBLE tokenomics (% burned, % staked, APY)
```

New surfaces (fish raffle, fishing competition, gacha, item drops, etc.) will slot in as sibling `references/<feature>/` subdirectories without touching existing docs.

## Links

- Game: https://cat.town
- Staking UI (Wealth & Whiskers Bank): https://cat.town/bank
- Public docs: https://docs.cat.town
- Chain: Base mainnet (8453)

## License

MIT
