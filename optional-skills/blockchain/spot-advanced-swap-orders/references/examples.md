# Examples

These are mock final relay payloads.
Copy the nearest shape, then replace addresses, amounts, timing, and signature.
Mix limit, trigger, and delay fields as needed.

## Limit Order

```json
{
  "order": {
    "permitted": {
      "token": "0x1111111111111111111111111111111111111111",
      "amount": "1000000"
    },
    "spender": "0x000000b33fE4fB9d999Dd684F79b110731c3d000",
    "nonce": "1712345601",
    "deadline": "1712345901",
    "witness": {
      "reactor": "0x000000b33fE4fB9d999Dd684F79b110731c3d000",
      "executor": "0x000642A0966d9bd49870D9519f76b5cf823f3000",
      "exchange": {
        "adapter": "0x9999999999999999999999999999999999999999",
        "ref": "0x0000000000000000000000000000000000000000",
        "share": 0,
        "data": "0x"
      },
      "swapper": "0x2222222222222222222222222222222222222222",
      "nonce": "1712345601",
      "start": "1712345601",
      "deadline": "1712345901",
      "chainid": 42161,
      "exclusivity": 0,
      "epoch": 0,
      "slippage": 500,
      "freshness": 50,
      "input": {
        "token": "0x1111111111111111111111111111111111111111",
        "amount": "1000000",
        "maxAmount": "1000000"
      },
      "output": {
        "token": "0x3333333333333333333333333333333333333333",
        "limit": "250000000000000",
        "triggerLower": "0",
        "triggerUpper": "0",
        "recipient": "0x2222222222222222222222222222222222222222"
      }
    }
  },
  "signature": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb1b"
}
```

## Stop-Loss Order

```json
{
  "order": {
    "permitted": {
      "token": "0x4444444444444444444444444444444444444444",
      "amount": "5000000000000000000"
    },
    "spender": "0x000000b33fE4fB9d999Dd684F79b110731c3d000",
    "nonce": "1712346602",
    "deadline": "1712347202",
    "witness": {
      "reactor": "0x000000b33fE4fB9d999Dd684F79b110731c3d000",
      "executor": "0x000642A0966d9bd49870D9519f76b5cf823f3000",
      "exchange": {
        "adapter": "0x9999999999999999999999999999999999999999",
        "ref": "0x0000000000000000000000000000000000000000",
        "share": 0,
        "data": "0x"
      },
      "swapper": "0x5555555555555555555555555555555555555555",
      "nonce": "1712346602",
      "start": "1712346602",
      "deadline": "1712347202",
      "chainid": 1,
      "exclusivity": 0,
      "epoch": 0,
      "slippage": 500,
      "freshness": 50,
      "input": {
        "token": "0x4444444444444444444444444444444444444444",
        "amount": "5000000000000000000",
        "maxAmount": "5000000000000000000"
      },
      "output": {
        "token": "0x6666666666666666666666666666666666666666",
        "limit": "0",
        "triggerLower": "9000000000000",
        "triggerUpper": "0",
        "recipient": "0x5555555555555555555555555555555555555555"
      }
    }
  },
  "signature": "0xccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd1c"
}
```

## TWAP Order

```json
{
  "order": {
    "permitted": {
      "token": "0x7777777777777777777777777777777777777777",
      "amount": "12000000"
    },
    "spender": "0x000000b33fE4fB9d999Dd684F79b110731c3d000",
    "nonce": "1712347603",
    "deadline": "1712348503",
    "witness": {
      "reactor": "0x000000b33fE4fB9d999Dd684F79b110731c3d000",
      "executor": "0x000642A0966d9bd49870D9519f76b5cf823f3000",
      "exchange": {
        "adapter": "0x9999999999999999999999999999999999999999",
        "ref": "0x0000000000000000000000000000000000000000",
        "share": 0,
        "data": "0x"
      },
      "swapper": "0x8888888888888888888888888888888888888888",
      "nonce": "1712347603",
      "start": "1712347603",
      "deadline": "1712348503",
      "chainid": 8453,
      "exclusivity": 0,
      "epoch": 300,
      "slippage": 500,
      "freshness": 50,
      "input": {
        "token": "0x7777777777777777777777777777777777777777",
        "amount": "3000000",
        "maxAmount": "12000000"
      },
      "output": {
        "token": "0x9999999999999999999999999999999999999999",
        "limit": "0",
        "triggerLower": "0",
        "triggerUpper": "0",
        "recipient": "0x8888888888888888888888888888888888888888"
      }
    }
  },
  "signature": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1b"
}
```

If a signer returns `{ "r": "...", "s": "...", "v": "..." }` instead of one full signature string, send that object unchanged in the same `signature` field.
