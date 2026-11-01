# Square OAuth Scopes

This document lists all available Square OAuth scopes and what they enable.

## Scopes Used by This Skill

| Scope | What it allows |
|-------|----------------|
| `ITEMS_READ` | Read catalog items, variations, categories, discounts, taxes |
| `ITEMS_WRITE` | Create and update catalog items |
| `INVENTORY_READ` | Read inventory counts and changes |
| `INVENTORY_WRITE` | Adjust inventory counts |
| `MERCHANT_PROFILE_READ` | Read business name, address, owner info |
| `CUSTOMERS_READ` | Read customer profiles |
| `CUSTOMERS_WRITE` | Create and update customers |
| `ORDERS_READ` | Read order data |
| `ORDERS_WRITE` | Create orders |
| `LOCATION_READ` | Read location information |

## All Available Scopes

### Payments
- `PAYMENTS_READ` — Accept and process payments
- `PAYMENTS_WRITE` — Make payments on behalf of the seller
- `PAYMENTS_WRITE_SHARED` — Take payments in your own app
- `PAYMENTS_WRITE_FORMER_CARD_PROCESSING` — Process card-present transactions

### Checkout
- `CHECKOUTS_READ` — Read checkout session data
- `CHECKOUTS_WRITE` — Create and manage checkout sessions

### Inventory
- `INVENTORY_READ` — View inventory counts and history
- `INVENTORY_WRITE` — Adjust inventory counts

### Catalog
- `ITEMS_READ` — Read catalog items and related data
- `ITEMS_WRITE` — Create and update catalog items

### Customers
- `CUSTOMERS_READ` — Read customer profiles
- `CUSTOMERS_WRITE` — Create and update customer profiles
- `CUSTOMER_GROUPS_READ` — Read customer groups
- `CUSTOMER_GROUPS_WRITE` — Manage customer groups
- `CUSTOMER_SEGMENTS_READ` — Read customer segments
- `LOYALTY_READ` — Read loyalty accounts and rewards
- `LOYALTY_WRITE` — Manage loyalty programs

### Orders
- `ORDERS_READ` — Read order data
- `ORDERS_WRITE` — Create and modify orders

### Payments & Deposits
- `BANK_ACCOUNTS_READ` — Read linked bank accounts
- `DEPOSITS_READ` — Read payout deposits
- `PAYOUTS_READ` — Read payout records

### Team
- `TIMECARDS_READ` — Read employee time cards
- `TIMECARDS_SETTINGS_READ` — Read time card settings
- `TIMECARDS_SETTINGS_WRITE` — Modify time card settings
- `EMPLOYEES_READ` — Read employee profiles
- `EMPLOYEES_WRITE` — Manage employee profiles
- `TEAM_READ` — Read team member info
- `TEAM_WRITE` — Manage team members
- `LABOR_READ` — Read labor and scheduling data
- `LABOR_WRITE` — Manage labor and scheduling

### Other
- `LOCATION_READ` — Read location details
- `MERCHANT_PROFILE_READ` — Read merchant/business profile
- `MERCHANT_PROFILE_WRITE` — Update business profile
- `ITEMS_READ` — Catalog read (also covers the deprecated Items API)
- `ITEMS_WRITE` — Catalog write
- `NOTES_READ` — Read seller notes
- `NOTES_WRITE` — Create and manage seller notes
- `GIFTCARDS_READ` — Read gift card data
- `GIFTCARDS_WRITE` — Create and modify gift cards
- `ACCOUNT_READ` — Read account and subscription info
- `ACCOUNT_WRITE` — Update account settings

## Least-Privilege Setup

For inventory-only access:
- `ITEMS_READ`, `ITEMS_WRITE`, `INVENTORY_READ`, `INVENTORY_WRITE`, `LOCATION_READ`

For customer management only:
- `CUSTOMERS_READ`, `CUSTOMERS_WRITE`, `LOCATION_READ`

For full commerce (recommended for most use cases):
- All scopes listed in setup.py's SCOPES list
