# Stripe Trigger Reference

## Event Types

- [`charge.succeeded`](#chargesucceeded)
- [`invoice.paid`](#invoicepaid)
- [`subscription.created`](#subscriptioncreated)
- [`subscription.canceled`](#subscriptioncanceled)
- [`invoice.payment_failed`](#invoicepayment_failed)

---

## `charge.succeeded`

### Sample Payload

```json
{
  "request": {
    "idempotency_key": "32a66241-bfd1-495f-a2a9-caaed284c6fb",
    "id": "req_maDSxdcS2FKV9w"
  },
  "data": {
    "object": {
      "balance_transaction": "txn_3TlzbCRWzNe0GNGm0ojPWGxW",
      "billing_details": {
        "address": {}
      },
      "metadata": {},
      "livemode": false,
      "radar_options": {},
      "description": "Subscription creation",
      "fraud_details": {},
      "amount_refunded": 0,
      "receipt_url": "https://pay.stripe.com/receipts/invoices/CAcaFwoVYWNj...",
      "captured": true,
      "calculated_statement_descriptor": "ACME, INC.",
      "currency": "usd",
      "refunded": false,
      "id": "ch_3TlzbCRWzNe0GNGm0syyVUT2",
      "outcome": {
        "risk_level": "normal",
        "seller_message": "Payment complete.",
        "network_status": "approved_by_network",
        "type": "authorized",
        "risk_score": 52
      },
      "payment_method": "pm_1TlzbARWzNe0GNGmbqYsLHYu",
      "amount": 2000,
      "disputed": false,
      "created": 1782341330,
      "payment_method_details": {
        "type": "card",
        "card": {
          "country": "US",
          "last4": "4242",
          "funding": "credit",
          "regulated_status": "unregulated",
          "exp_month": 6,
          "exp_year": 2027,
          "overcapture": {
            "maximum_amount_capturable": 2000,
            "status": "unavailable"
          },
          "amount_authorized": 2000,
          "network": "visa",
          "network_token": {
            "used": false
          },
          "network_transaction_id": "727986839810999",
          "incremental_authorization": {
            "status": "unavailable"
          },
          "checks": {
            "cvc_check": "pass"
          },
          "extended_authorization": {
            "status": "disabled"
          },
          "authorization_code": "869540",
          "fingerprint": "KyTu5DgnPAlPPbBr",
          "brand": "visa",
          "multicapture": {
            "status": "unavailable"
          }
        }
      },
      "amount_captured": 2000,
      "application": "ca_00000000000000000000000000",
      "paid": true,
      "payment_intent": "pi_3TlzbCRWzNe0GNGm00yfBTIf",
      "invoice": "in_1TlzbBRWzNe0GNGmStbaJSGV",
      "object": "charge",
      "customer": "cus_UlWecDPcrrvdG2",
      "status": "succeeded"
    }
  },
  "livemode": false,
  "created": 1782341331,
  "context": "acct_00000000000000",
  "id": "evt_3TlzbCRWzNe0GNGm0O6lRqVg",
  "api_version": "2022-11-15",
  "type": "charge.succeeded",
  "account": "acct_00000000000000",
  "pending_webhooks": 2,
  "object": "event"
}
```

---

## `invoice.paid`

### Sample Payload

```json
{
  "request": {
    "idempotency_key": "32a66241-bfd1-495f-a2a9-caaed284c6fb",
    "id": "req_maDSxdcS2FKV9w"
  },
  "data": {
    "object": {
      "parent": {
        "type": "subscription_details",
        "subscription_details": {
          "metadata": {},
          "subscription": "sub_1TlzbBRWzNe0GNGmpgiGtqSY"
        }
      },
      "amount_paid": 2000,
      "amount_remaining": 0,
      "metadata": {},
      "livemode": false,
      "total_tax_amounts": [],
      "amount_shipping": 0,
      "issuer": {
        "type": "self"
      },
      "number": "TSNNVFLX-0001",
      "auto_advance": false,
      "discounts": [],
      "subtotal_excluding_tax": 2000,
      "period_start": 1782341329,
      "amount_due": 2000,
      "id": "in_1TlzbBRWzNe0GNGmStbaJSGV",
      "payment_settings": {},
      "paid_out_of_band": false,
      "created": 1782341329,
      "ending_balance": 0,
      "period_end": 1782341329,
      "webhooks_delivered_at": 1782341329,
      "subtotal": 2000,
      "subscription_details": {
        "metadata": {}
      },
      "customer_email": "customer-a@example.com",
      "payment_intent": "pi_3TlzbCRWzNe0GNGm00yfBTIf",
      "amount_overpaid": 0,
      "object": "invoice",
      "status": "paid",
      "invoice_pdf": "https://pay.stripe.com/invoice/acct_00000000000000/test_YWNj.../pdf?s=ap",
      "total_taxes": [],
      "subscription": "sub_1TlzbBRWzNe0GNGmpgiGtqSY",
      "starting_balance": 0,
      "total": 2000,
      "pre_payment_credit_notes_amount": 0,
      "total_pretax_credit_amounts": [],
      "account_name": "Acme, Inc.",
      "currency": "usd",
      "account_country": "US",
      "lines": {
        "has_more": false,
        "data": [
          {
            "parent": {
              "subscription_item_details": {
                "subscription": "sub_1TlzbBRWzNe0GNGmpgiGtqSY",
                "proration_details": {},
                "proration": false,
                "subscription_item": "si_UlWe8NWmBI15e1"
              },
              "type": "subscription_item_details"
            },
            "unit_amount_excluding_tax": "2000",
            "metadata": {},
            "livemode": false,
            "pretax_credit_amounts": [],
            "description": "1 × Starter Plan (at $20.00 / month)",
            "taxes": [],
            "subscription": "sub_1TlzbBRWzNe0GNGmpgiGtqSY",
            "type": "subscription",
            "discounts": [],
            "price": {
              "tax_behavior": "exclusive",
              "metadata": {},
              "product": "prod_SAoe9vYQVS3O4e",
              "livemode": false,
              "created": 1745275248,
              "recurring": {
                "interval_count": 1,
                "usage_type": "licensed",
                "interval": "month"
              },
              "active": true,
              "unit_amount_decimal": "2000",
              "billing_scheme": "per_unit",
              "unit_amount": 2000,
              "type": "recurring",
              "currency": "usd",
              "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
              "object": "price"
            },
            "subscription_item": "si_UlWe8NWmBI15e1",
            "amount_excluding_tax": 2000,
            "currency": "usd",
            "id": "il_1TlzbBRWzNe0GNGmm293tSOD",
            "discountable": true,
            "plan": {
              "interval_count": 1,
              "amount": 2000,
              "metadata": {},
              "product": "prod_SAoe9vYQVS3O4e",
              "livemode": false,
              "created": 1745275248,
              "amount_decimal": "2000",
              "active": true,
              "billing_scheme": "per_unit",
              "usage_type": "licensed",
              "currency": "usd",
              "interval": "month",
              "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
              "object": "plan"
            },
            "discount_amounts": [],
            "amount": 2000,
            "period": {
              "start": 1782341329,
              "end": 1784933329
            },
            "quantity": 1,
            "proration": false,
            "tax_amounts": [],
            "tax_rates": [],
            "quantity_decimal": "1",
            "proration_details": {},
            "subtotal": 2000,
            "invoice": "in_1TlzbBRWzNe0GNGmStbaJSGV",
            "pricing": {
              "type": "price_details",
              "unit_amount_decimal": "2000",
              "price_details": {
                "price": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
                "product": "prod_SAoe9vYQVS3O4e"
              }
            },
            "object": "line_item"
          }
        ],
        "total_count": 1,
        "url": "/v1/invoices/in_1TlzbBRWzNe0GNGmStbaJSGV/lines",
        "object": "list"
      },
      "total_discount_amounts": [],
      "attempt_count": 1,
      "default_tax_rates": [],
      "charge": "ch_3TlzbCRWzNe0GNGm0syyVUT2",
      "total_excluding_tax": 2000,
      "post_payment_credit_notes_amount": 0,
      "billing_reason": "subscription_create",
      "effective_at": 1782341329,
      "automatic_tax": {
        "enabled": false
      },
      "application": "ca_00000000000000000000000000",
      "collection_method": "charge_automatically",
      "customer_tax_ids": [],
      "hosted_invoice_url": "https://invoice.stripe.com/i/acct_00000000000000/test_YWNj...?s=ap",
      "paid": true,
      "attempted": true,
      "status_transitions": {
        "finalized_at": 1782341329,
        "paid_at": 1782341329
      },
      "customer": "cus_UlWecDPcrrvdG2",
      "customer_tax_exempt": "none"
    }
  },
  "livemode": false,
  "created": 1782341331,
  "context": "acct_00000000000000",
  "id": "evt_1TlzbERWzNe0GNGmi5dTac2W",
  "api_version": "2022-11-15",
  "type": "invoice.paid",
  "account": "acct_00000000000000",
  "pending_webhooks": 4,
  "object": "event"
}
```

---

## `subscription.created`

### Sample Payload

```json
{
  "request": {
    "idempotency_key": "32a66241-bfd1-495f-a2a9-caaed284c6fb",
    "id": "req_maDSxdcS2FKV9w"
  },
  "data": {
    "object": {
      "metadata": {},
      "billing_mode": {
        "type": "classic"
      },
      "cancel_at_period_end": false,
      "livemode": false,
      "cancellation_details": {},
      "discounts": [],
      "currency": "usd",
      "id": "sub_1TlzbBRWzNe0GNGmpgiGtqSY",
      "plan": {
        "interval_count": 1,
        "amount": 2000,
        "metadata": {},
        "product": "prod_SAoe9vYQVS3O4e",
        "livemode": false,
        "created": 1745275248,
        "amount_decimal": "2000",
        "active": true,
        "billing_scheme": "per_unit",
        "usage_type": "licensed",
        "currency": "usd",
        "interval": "month",
        "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
        "object": "plan"
      },
      "current_period_start": 1782341329,
      "start_date": 1782341329,
      "invoice_settings": {
        "issuer": {
          "type": "self"
        }
      },
      "payment_settings": {
        "save_default_payment_method": "off"
      },
      "default_tax_rates": [],
      "latest_invoice": "in_1TlzbBRWzNe0GNGmStbaJSGV",
      "quantity": 1,
      "created": 1782341329,
      "managed_payments": {
        "enabled": false
      },
      "trial_settings": {
        "end_behavior": {
          "missing_payment_method": "create_invoice"
        }
      },
      "current_period_end": 1784933329,
      "billing_cycle_anchor": 1782341329,
      "automatic_tax": {
        "enabled": false
      },
      "application": "ca_00000000000000000000000000",
      "collection_method": "charge_automatically",
      "items": {
        "has_more": false,
        "data": [
          {
            "metadata": {},
            "quantity": 1,
            "discounts": [],
            "created": 1782341329,
            "price": {
              "tax_behavior": "exclusive",
              "metadata": {},
              "product": "prod_SAoe9vYQVS3O4e",
              "livemode": false,
              "created": 1745275248,
              "recurring": {
                "interval_count": 1,
                "usage_type": "licensed",
                "interval": "month"
              },
              "active": true,
              "unit_amount_decimal": "2000",
              "billing_scheme": "per_unit",
              "unit_amount": 2000,
              "type": "recurring",
              "currency": "usd",
              "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
              "object": "price"
            },
            "id": "si_UlWe8NWmBI15e1",
            "subscription": "sub_1TlzbBRWzNe0GNGmpgiGtqSY",
            "current_period_end": 1784933329,
            "plan": {
              "interval_count": 1,
              "amount": 2000,
              "metadata": {},
              "product": "prod_SAoe9vYQVS3O4e",
              "livemode": false,
              "created": 1745275248,
              "amount_decimal": "2000",
              "active": true,
              "billing_scheme": "per_unit",
              "usage_type": "licensed",
              "currency": "usd",
              "interval": "month",
              "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
              "object": "plan"
            },
            "current_period_start": 1782341329,
            "object": "subscription_item",
            "tax_rates": []
          }
        ],
        "total_count": 1,
        "url": "/v1/subscription_items?subscription=sub_1TlzbBRWzNe0GNGmpgiGtqSY",
        "object": "list"
      },
      "object": "subscription",
      "customer": "cus_UlWecDPcrrvdG2",
      "status": "active"
    }
  },
  "livemode": false,
  "created": 1782341331,
  "context": "acct_00000000000000",
  "id": "evt_1TlzbERWzNe0GNGmryXuTBGH",
  "api_version": "2022-11-15",
  "type": "customer.subscription.created",
  "account": "acct_00000000000000",
  "pending_webhooks": 4,
  "object": "event"
}
```

---

## `subscription.canceled`

### Sample Payload

```json
{
  "request": {
    "id": "req_Po036ftLjgzm98"
  },
  "data": {
    "object": {
      "metadata": {},
      "billing_mode": {
        "type": "classic"
      },
      "cancel_at_period_end": false,
      "livemode": false,
      "canceled_at": 1782341334,
      "cancellation_details": {
        "reason": "cancellation_requested"
      },
      "discounts": [],
      "currency": "usd",
      "id": "sub_1TlzbBRWzNe0GNGmpgiGtqSY",
      "plan": {
        "interval_count": 1,
        "amount": 2000,
        "metadata": {},
        "product": "prod_SAoe9vYQVS3O4e",
        "livemode": false,
        "created": 1745275248,
        "amount_decimal": "2000",
        "active": true,
        "billing_scheme": "per_unit",
        "usage_type": "licensed",
        "currency": "usd",
        "interval": "month",
        "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
        "object": "plan"
      },
      "current_period_start": 1782341329,
      "start_date": 1782341329,
      "invoice_settings": {
        "issuer": {
          "type": "self"
        }
      },
      "payment_settings": {
        "save_default_payment_method": "off"
      },
      "default_tax_rates": [],
      "latest_invoice": "in_1TlzbBRWzNe0GNGmStbaJSGV",
      "quantity": 1,
      "created": 1782341329,
      "managed_payments": {
        "enabled": false
      },
      "trial_settings": {
        "end_behavior": {
          "missing_payment_method": "create_invoice"
        }
      },
      "current_period_end": 1784933329,
      "billing_cycle_anchor": 1782341329,
      "automatic_tax": {
        "enabled": false
      },
      "application": "ca_00000000000000000000000000",
      "collection_method": "charge_automatically",
      "items": {
        "has_more": false,
        "data": [
          {
            "metadata": {},
            "quantity": 1,
            "discounts": [],
            "created": 1782341329,
            "price": {
              "tax_behavior": "exclusive",
              "metadata": {},
              "product": "prod_SAoe9vYQVS3O4e",
              "livemode": false,
              "created": 1745275248,
              "recurring": {
                "interval_count": 1,
                "usage_type": "licensed",
                "interval": "month"
              },
              "active": true,
              "unit_amount_decimal": "2000",
              "billing_scheme": "per_unit",
              "unit_amount": 2000,
              "type": "recurring",
              "currency": "usd",
              "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
              "object": "price"
            },
            "id": "si_UlWe8NWmBI15e1",
            "subscription": "sub_1TlzbBRWzNe0GNGmpgiGtqSY",
            "current_period_end": 1784933329,
            "plan": {
              "interval_count": 1,
              "amount": 2000,
              "metadata": {},
              "product": "prod_SAoe9vYQVS3O4e",
              "livemode": false,
              "created": 1745275248,
              "amount_decimal": "2000",
              "active": true,
              "billing_scheme": "per_unit",
              "usage_type": "licensed",
              "currency": "usd",
              "interval": "month",
              "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
              "object": "plan"
            },
            "current_period_start": 1782341329,
            "object": "subscription_item",
            "tax_rates": []
          }
        ],
        "total_count": 1,
        "url": "/v1/subscription_items?subscription=sub_1TlzbBRWzNe0GNGmpgiGtqSY",
        "object": "list"
      },
      "ended_at": 1782341334,
      "object": "subscription",
      "customer": "cus_UlWecDPcrrvdG2",
      "status": "canceled"
    }
  },
  "livemode": false,
  "created": 1782341334,
  "context": "acct_00000000000000",
  "id": "evt_1TlzbGRWzNe0GNGmPUToohOW",
  "api_version": "2022-11-15",
  "type": "customer.subscription.deleted",
  "account": "acct_00000000000000",
  "pending_webhooks": 4,
  "object": "event"
}
```

---

## `invoice.payment_failed`

### Sample Payload

```json
{
  "request": {
    "idempotency_key": "8bbe2da0-0428-454d-997c-fe7e19a7615e",
    "id": "req_nxsqWBWDQfRaOg"
  },
  "data": {
    "object": {
      "parent": {
        "type": "subscription_details",
        "subscription_details": {
          "metadata": {},
          "subscription": "sub_1TlzbIRWzNe0GNGmX7mm9rvA"
        }
      },
      "amount_paid": 0,
      "amount_remaining": 2000,
      "metadata": {},
      "livemode": false,
      "total_tax_amounts": [],
      "amount_shipping": 0,
      "issuer": {
        "type": "self"
      },
      "auto_advance": true,
      "discounts": [],
      "subtotal_excluding_tax": 2000,
      "period_start": 1782341336,
      "amount_due": 2000,
      "id": "in_1TlzbIRWzNe0GNGm6BCH1QVe",
      "payment_settings": {},
      "paid_out_of_band": false,
      "created": 1782341336,
      "ending_balance": 0,
      "period_end": 1782341336,
      "webhooks_delivered_at": 1782341336,
      "subtotal": 2000,
      "subscription_details": {
        "metadata": {}
      },
      "customer_email": "customer-b@example.com",
      "payment_intent": "pi_3TlzbJRWzNe0GNGm1S2FtbF6",
      "amount_overpaid": 0,
      "object": "invoice",
      "status": "open",
      "invoice_pdf": "https://pay.stripe.com/invoice/acct_00000000000000/test_YWNj.../pdf?s=ap",
      "total_taxes": [],
      "subscription": "sub_1TlzbIRWzNe0GNGmX7mm9rvA",
      "starting_balance": 0,
      "total": 2000,
      "pre_payment_credit_notes_amount": 0,
      "total_pretax_credit_amounts": [],
      "account_name": "Acme, Inc.",
      "currency": "usd",
      "account_country": "US",
      "lines": {
        "has_more": false,
        "data": [
          {
            "parent": {
              "subscription_item_details": {
                "subscription": "sub_1TlzbIRWzNe0GNGmX7mm9rvA",
                "proration_details": {},
                "proration": false,
                "subscription_item": "si_UlWeMK9NlXKmGL"
              },
              "type": "subscription_item_details"
            },
            "unit_amount_excluding_tax": "2000",
            "metadata": {},
            "livemode": false,
            "pretax_credit_amounts": [],
            "description": "1 × Starter Plan (at $20.00 / month)",
            "taxes": [],
            "subscription": "sub_1TlzbIRWzNe0GNGmX7mm9rvA",
            "type": "subscription",
            "discounts": [],
            "price": {
              "tax_behavior": "exclusive",
              "metadata": {},
              "product": "prod_SAoe9vYQVS3O4e",
              "livemode": false,
              "created": 1745275248,
              "recurring": {
                "interval_count": 1,
                "usage_type": "licensed",
                "interval": "month"
              },
              "active": true,
              "unit_amount_decimal": "2000",
              "billing_scheme": "per_unit",
              "unit_amount": 2000,
              "type": "recurring",
              "currency": "usd",
              "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
              "object": "price"
            },
            "subscription_item": "si_UlWeMK9NlXKmGL",
            "amount_excluding_tax": 2000,
            "currency": "usd",
            "id": "il_1TlzbIRWzNe0GNGm2Ym33uo5",
            "discountable": true,
            "plan": {
              "interval_count": 1,
              "amount": 2000,
              "metadata": {},
              "product": "prod_SAoe9vYQVS3O4e",
              "livemode": false,
              "created": 1745275248,
              "amount_decimal": "2000",
              "active": true,
              "billing_scheme": "per_unit",
              "usage_type": "licensed",
              "currency": "usd",
              "interval": "month",
              "id": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
              "object": "plan"
            },
            "discount_amounts": [],
            "amount": 2000,
            "period": {
              "start": 1782341336,
              "end": 1784933336
            },
            "quantity": 1,
            "proration": false,
            "tax_amounts": [],
            "tax_rates": [],
            "quantity_decimal": "1",
            "proration_details": {},
            "subtotal": 2000,
            "invoice": "in_1TlzbIRWzNe0GNGm6BCH1QVe",
            "pricing": {
              "type": "price_details",
              "unit_amount_decimal": "2000",
              "price_details": {
                "price": "price_1RGT1ARWzNe0GNGmI3Ff0S6u",
                "product": "prod_SAoe9vYQVS3O4e"
              }
            },
            "object": "line_item"
          }
        ],
        "total_count": 1,
        "url": "/v1/invoices/in_1TlzbIRWzNe0GNGm6BCH1QVe/lines",
        "object": "list"
      },
      "total_discount_amounts": [],
      "attempt_count": 1,
      "default_tax_rates": [],
      "charge": "ch_3TlzbJRWzNe0GNGm1TIbZcLS",
      "total_excluding_tax": 2000,
      "post_payment_credit_notes_amount": 0,
      "billing_reason": "subscription_create",
      "effective_at": 1782341336,
      "automatic_tax": {
        "enabled": false
      },
      "application": "ca_00000000000000000000000000",
      "collection_method": "charge_automatically",
      "customer_tax_ids": [],
      "hosted_invoice_url": "https://invoice.stripe.com/i/acct_00000000000000/test_YWNj...?s=ap",
      "paid": false,
      "attempted": true,
      "status_transitions": {
        "finalized_at": 1782341336
      },
      "customer": "cus_UlWeN7c8OHpXNO",
      "customer_tax_exempt": "none"
    }
  },
  "livemode": false,
  "created": 1782341339,
  "context": "acct_00000000000000",
  "id": "evt_1TlzbMRWzNe0GNGmZPzZQNH7",
  "api_version": "2022-11-15",
  "type": "invoice.payment_failed",
  "account": "acct_00000000000000",
  "pending_webhooks": 2,
  "object": "event"
}
```