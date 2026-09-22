# Synthetic sample input

> **SYNTHETIC FIXTURE — not real issuer terms or account data.** Rates, caps, credits, and card nicknames are invented for offline calculation testing.

Purchase amount: 100.00 fictional dollars. User values one reward unit at one cent.

```csv
nickname,reward_rate_percent,cap_remaining,credit_remaining
Everyday,2,1000,0
Travel,3,20,5
```

Expected task: calculate both values with the helper, explain the cap and synthetic credit, and state that a real recommendation requires current official issuer verification. Do not ask for any account secret.
