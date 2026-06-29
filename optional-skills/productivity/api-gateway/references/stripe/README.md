# Stripe Routing Reference

> **Safety:** All write operations (POST, PUT, PATCH, DELETE) require explicit user confirmation before execution. Verify the target resource and intended effect with the user first. See the main [SKILL.md](../SKILL.md#security--permissions) for full security policy.

**App name:** `stripe`
**Base URL proxied:** `api.stripe.com`

## API Path Pattern

```
/stripe/v1/{endpoint}
```

## Common Endpoints

### Customers

#### List Customers
```bash
GET /stripe/v1/customers?limit=10
```

Example:

```bash
maton stripe customer list -L 10
```

#### Get Customer
```bash
GET /stripe/v1/customers/{customerId}
```

Example:

```bash
maton stripe customer view {customerId}
```

#### Create Customer
```bash
POST /stripe/v1/customers
Content-Type: application/x-www-form-urlencoded

email=customer@example.com&name=John%20Doe&description=New%20customer
```

Example:

```bash
maton stripe customer create --email customer@example.com --name 'John Doe'
```

#### Update Customer
```bash
POST /stripe/v1/customers/{customerId}
Content-Type: application/x-www-form-urlencoded

email=newemail@example.com
```

Example:

```bash
maton stripe customer update {customerId} --email newemail@example.com
```

### Products

#### List Products
```bash
GET /stripe/v1/products?limit=10&active=true
```

Example:

```bash
maton stripe product list -L 10
```

#### Create Product
```bash
POST /stripe/v1/products
Content-Type: application/x-www-form-urlencoded

name=Premium%20Plan&description=Monthly%20subscription
```

Example:

```bash
maton stripe product create --name 'Premium Plan' --description 'Monthly subscription'
```

### Prices

#### List Prices
```bash
GET /stripe/v1/prices?limit=10&active=true
```

Example:

```bash
maton stripe price list -L 10
```

#### Create Price
```bash
POST /stripe/v1/prices
Content-Type: application/x-www-form-urlencoded

unit_amount=1999&currency=usd&product=prod_XXX&recurring[interval]=month
```

Example:

```bash
maton stripe price create --product prod_XXX --unit-amount 1999 --currency usd --recurring-interval month
```

### Subscriptions

#### List Subscriptions
```bash
GET /stripe/v1/subscriptions?limit=10&status=active
```

Example:

```bash
maton stripe subscription list -L 10
```

#### Get Subscription
```bash
GET /stripe/v1/subscriptions/{subscriptionId}
```

Example:

```bash
maton stripe subscription view {subscriptionId}
```

#### Create Subscription
```bash
POST /stripe/v1/subscriptions
Content-Type: application/x-www-form-urlencoded

customer=cus_XXX&items[0][price]=price_XXX
```

Example:

```bash
maton stripe subscription create --customer cus_XXX --price price_XXX
```

#### Cancel Subscription
```bash
DELETE /stripe/v1/subscriptions/{subscriptionId}
```

Example:

```bash
maton stripe subscription cancel {subscriptionId}
```

### Invoices

#### List Invoices
```bash
GET /stripe/v1/invoices?limit=10&customer=cus_XXX
```

Example:

```bash
maton stripe invoice list -L 10
```

#### Get Invoice
```bash
GET /stripe/v1/invoices/{invoiceId}
```

Example:

```bash
maton stripe invoice view {invoiceId}
```

### Charges

#### List Charges
```bash
GET /stripe/v1/charges?limit=10
```

Example:

```bash
maton stripe charge list -L 10
```

### Payment Intents

#### Create Payment Intent
```bash
POST /stripe/v1/payment_intents
Content-Type: application/x-www-form-urlencoded

amount=1999&currency=usd&customer=cus_XXX
```

Example:

```bash
maton stripe payment create --amount 1999 --currency usd --customer cus_XXX
```

### Balance

#### Get Balance
```bash
GET /stripe/v1/balance
```

Example:

```bash
maton stripe balance
```

### Events

#### List Events
```bash
GET /stripe/v1/events?limit=10&type=customer.created
```

### Payment Methods

#### List Payment Methods
```bash
GET /stripe/v1/payment_methods?customer=cus_XXX&type=card
```

Example:

```bash
maton stripe payment-method list --customer cus_XXX --type card
```

#### Attach Payment Method
```bash
POST /stripe/v1/payment_methods/{paymentMethodId}/attach
Content-Type: application/x-www-form-urlencoded

customer=cus_XXX
```

Example:

```bash
maton stripe payment-method attach {paymentMethodId} --customer cus_XXX
```

#### Detach Payment Method
```bash
POST /stripe/v1/payment_methods/{paymentMethodId}/detach
```

Example:

```bash
maton stripe payment-method detach {paymentMethodId}
```

### Coupons

#### List Coupons
```bash
GET /stripe/v1/coupons?limit=10
```

Example:

```bash
maton stripe coupon list -L 10
```

#### Create Coupon
```bash
POST /stripe/v1/coupons
Content-Type: application/x-www-form-urlencoded

percent_off=25&duration=once
```

Example:

```bash
maton stripe coupon create --percent-off 25 --duration once
```

#### Delete Coupon
```bash
DELETE /stripe/v1/coupons/{couponId}
```

Example:

```bash
maton stripe coupon delete {couponId}
```

### Refunds

#### List Refunds
```bash
GET /stripe/v1/refunds?limit=10
```

Example:

```bash
maton stripe refund list -L 10
```

#### Create Refund
```bash
POST /stripe/v1/refunds
Content-Type: application/x-www-form-urlencoded

charge=ch_XXX&amount=1000
```

Example:

```bash
maton stripe refund create --charge ch_XXX --amount 1000
```

## Pagination

Stripe uses cursor-based pagination with `starting_after` and `ending_before`:

```bash
GET /stripe/v1/customers?limit=10&starting_after=cus_XXX
```

Example:

```bash
maton stripe customer list -L 10 --starting-after cus_XXX
```

Use the last item's ID as `starting_after` for the next page.

## Notes

- Stripe API uses form-urlencoded data for POST requests
- IDs are prefixed: `cus_` (customer), `sub_` (subscription), `prod_` (product), `price_` (price), `in_` (invoice), `pi_` (payment intent)
- Amounts are in cents (1999 = $19.99)
- Use `expand[]` parameter to include related objects; for list endpoints use `expand[]=data.{field}` (e.g., `expand[]=data.customer`)
- List endpoints support pagination with `starting_after` and `ending_before`
- Delete returns `{id, deleted: true}` on success
- Products with prices cannot be deleted, only archived (`active=false`)

## Resources

- [API Overview](https://docs.stripe.com/api)
- [List Customers](https://docs.stripe.com/api/customers/list.md)
- [Get Customer](https://docs.stripe.com/api/customers/retrieve.md)
- [Create Customer](https://docs.stripe.com/api/customers/create.md)
- [Update Customer](https://docs.stripe.com/api/customers/update.md)
- [Delete Customer](https://docs.stripe.com/api/customers/delete.md)
- [Search Customers](https://docs.stripe.com/api/customers/search.md)
- [List Products](https://docs.stripe.com/api/products/list.md)
- [Get Product](https://docs.stripe.com/api/products/retrieve.md)
- [Create Product](https://docs.stripe.com/api/products/create.md)
- [Update Product](https://docs.stripe.com/api/products/update.md)
- [Delete Product](https://docs.stripe.com/api/products/delete.md)
- [Search Products](https://docs.stripe.com/api/products/search.md)
- [List Prices](https://docs.stripe.com/api/prices/list.md)
- [Get Price](https://docs.stripe.com/api/prices/retrieve.md)
- [Create Price](https://docs.stripe.com/api/prices/create.md)
- [Update Price](https://docs.stripe.com/api/prices/update.md)
- [Search Prices](https://docs.stripe.com/api/prices/search.md)
- [List Subscriptions](https://docs.stripe.com/api/subscriptions/list.md)
- [Get Subscription](https://docs.stripe.com/api/subscriptions/retrieve.md)
- [Create Subscription](https://docs.stripe.com/api/subscriptions/create.md)
- [Update Subscription](https://docs.stripe.com/api/subscriptions/update.md)
- [Cancel Subscription](https://docs.stripe.com/api/subscriptions/cancel.md)
- [Resume Subscription](https://docs.stripe.com/api/subscriptions/resume.md)
- [Search Subscriptions](https://docs.stripe.com/api/subscriptions/search.md)
- [List Invoices](https://docs.stripe.com/api/invoices/list.md)
- [Get Invoice](https://docs.stripe.com/api/invoices/retrieve.md)
- [Create Invoice](https://docs.stripe.com/api/invoices/create.md)
- [Update Invoice](https://docs.stripe.com/api/invoices/update.md)
- [Delete Invoice](https://docs.stripe.com/api/invoices/delete.md)
- [Finalize Invoice](https://docs.stripe.com/api/invoices/finalize.md)
- [Pay Invoice](https://docs.stripe.com/api/invoices/pay.md)
- [Send Invoice](https://docs.stripe.com/api/invoices/send.md)
- [Void Invoice](https://docs.stripe.com/api/invoices/void.md)
- [Search Invoices](https://docs.stripe.com/api/invoices/search.md)
- [List Charges](https://docs.stripe.com/api/charges/list.md)
- [Get Charge](https://docs.stripe.com/api/charges/retrieve.md)
- [Create Charge](https://docs.stripe.com/api/charges/create.md)
- [Update Charge](https://docs.stripe.com/api/charges/update.md)
- [Capture Charge](https://docs.stripe.com/api/charges/capture.md)
- [Search Charges](https://docs.stripe.com/api/charges/search.md)
- [List Payment Intents](https://docs.stripe.com/api/payment_intents/list.md)
- [Get Payment Intent](https://docs.stripe.com/api/payment_intents/retrieve.md)
- [Create Payment Intent](https://docs.stripe.com/api/payment_intents/create.md)
- [Update Payment Intent](https://docs.stripe.com/api/payment_intents/update.md)
- [Confirm Payment Intent](https://docs.stripe.com/api/payment_intents/confirm.md)
- [Capture Payment Intent](https://docs.stripe.com/api/payment_intents/capture.md)
- [Cancel Payment Intent](https://docs.stripe.com/api/payment_intents/cancel.md)
- [Search Payment Intents](https://docs.stripe.com/api/payment_intents/search.md)
- [Get Balance](https://docs.stripe.com/api/balance/balance_retrieve.md)
- [List Balance Transactions](https://docs.stripe.com/api/balance_transactions/list.md)
- [Get Balance Transaction](https://docs.stripe.com/api/balance_transactions/retrieve.md)
- [List Events](https://docs.stripe.com/api/events/list.md)
- [Get Event](https://docs.stripe.com/api/events/retrieve.md)
- [Pagination](https://docs.stripe.com/api/pagination.md)
- [Expanding Responses](https://docs.stripe.com/api/expanding_objects.md)
- [LLM Reference](https://docs.stripe.com/llms.txt)
- [Maton CLI Manual](https://cli.maton.ai/manual)