# GMC Misrepresentation Policy Baseline

Reference document for `shopify-gmc-misrepresentation-auditor`. Load this before scoring, classifying findings, or generating the report.

## Official Policy Sources

- Shopping ads Misrepresentation policy: https://support.google.com/merchants/answer/6150127
- Free listings Misrepresentation policy: https://support.google.com/merchants/answer/12079606
- Landing page requirements: https://support.google.com/merchants/answer/4752265
- Checkout requirements: https://support.google.com/merchants/answer/9158778
- Product data specification: https://support.google.com/merchants/answer/7052112
- Building trust with your customers: https://support.google.com/merchants/answer/188484

## Four Policy Buckets

### 1. Unacceptable Business Practices

Google prohibits:
- Fake identity, business name, or contact information
- Impersonating other brands or businesses
- Claiming certifications or partnerships that don't exist
- Offering products you don't have or can't deliver
- Phishing techniques to gather user information
- Denying return/refund despite having a clear policy that allows it

**Egregious violations** (immediate suspension, no warning):
- Cloaking or IP-based redirects that show different content to Google vs users
- False Google association claims ("Certified by Google", "Google Partner")
- Fabricated reviews or trust badges from non-existent organizations
- Unsupported medical/regulatory claims ("FDA approved", "clinically proven cure")
- Non-functional return/refund process when policy states returns are accepted

### 2. Misleading or Unrealistic Offers

Google prohibits:
- False claims or improbable results presented as likely outcomes
- Falsely implying affiliation with or endorsement by another individual/organization
- Misleading use of official government sites, stamps, seals, or agency names
- Harmful health claims contradicting authoritative scientific consensus
- Fake urgency (evergreen countdown timers, static "only X left" messages)
- Compare-at prices that misrepresent the original price history

### 3. Omission of Relevant Information

Google prohibits:
- Failure to clearly disclose the full price before purchase (hidden fees)
- Missing or unclear return/refund/cancellation policy
- Missing shipping information
- Omitting material information when promoting charitable or political content
- Pricing that depends on undisclosed conditions (membership fees, contracts)

### 4. Unavailable Offers

Google prohibits:
- Promoting products that aren't stocked
- Promoting deals that are no longer active
- Add-to-cart or checkout that doesn't work
- JSON-LD availability claiming InStock when product is actually sold out

## Severity Classification

### Critical (SeverityWeight = 25)
Findings that directly match a documented GMC policy violation and have high confidence evidence:
- Price mismatch between DOM, JSON-LD, and checkout
- Product shown as InStock in JSON-LD but add-to-cart is disabled
- Unsupported medical/health/efficacy claims in product descriptions
- Missing return/refund policy page (404 or no policy found)
- Liquid template errors visible on product pages
- robots.txt blocking Googlebot on product pages
- SSL certificate invalid or expired

### High (SeverityWeight = 12)
Findings that are strong risk signals with medium-to-high confidence:
- Missing About Us page or page with < 100 words
- Missing contact page or no contact information
- Countdown timers that reset on page refresh (evergreen)
- "Only X left" messages that don't change across fetches
- Badge/certification claims with broken or missing verification links
- "As Seen On" logos without verifiable media coverage
- aggregateRating in JSON-LD without visible review widget on page
- Policy pages returning 4xx/5xx errors
- Broken internal links (4xx/5xx)

### Medium (SeverityWeight = 5)
Findings that are risk signals requiring merchant verification:
- Compare-at / was pricing present (cannot verify historical price)
- Discount copy that persists across two fetches (may be evergreen)
- Visitor counter or "X bought recently" widgets (may be static)
- Missing GTIN/MPN/brand in product JSON-LD
- Empty schema fields (material, color, size)
- Stale year (N-2 or older) in meta description or product description
- Review count inconsistency between JSON-LD and DOM widget

### Low (SeverityWeight = 2)
Minor quality issues that don't directly trigger GMC policy:
- Missing product dimensions in schema
- Tab/accordion content not indexed in initial HTML
- Off-domain images (CDN delivery is standard)

## Confidence Levels

- **High (1.0)**: DOM evidence + JSON-LD evidence + interaction test all agree
- **Medium (0.7)**: DOM evidence + JSON-LD evidence agree, no interaction test
- **Low (0.4)**: DOM evidence only, no JSON-LD confirmation
- **Heuristic (0.75 EvidenceFactor)**: Regex pattern match only, no structural confirmation

## False Positive Risk Management

| Check | FP Risk | Mitigation |
|-------|---------|------------|
| Compare-at price flagging | Seasonal clearance may have genuine > 50% off | Label as "risk signal", not "fake discount" |
| Award claim detection | Industry-standard awards are legitimate | Add exclusion list for known industry awards |
| Empty schema fields | Theme may not expose all fields | Note as quality gap, not violation |
| Off-domain images | CDN delivery is standard | Only as trust signal, not standalone violation |
| Donation/eco claims | May link to real verification pages | HEAD-check linked URLs; lower severity if pages exist |

## Shopify-Specific Known Issues

### High-Frequency Suspension Causes (from Shopify community 2024-2025)

1. **Address mismatch**: GMC account address vs website footer address — even "Inc." vs no "Inc." has triggered reviews. Cannot be detected by crawling; included in MC-01 checklist.

2. **Feed vs landing page price mismatch**: Most common technical cause. Cannot be verified without feed access; included in MC-02 checklist.

3. **Liquid template errors**: `Liquid error: ...` visible in page HTML is a known automatic suspension trigger. Detectable by crawling.

4. **Fake business address**: Address belonging to a shipping/mailing service rather than actual business. Partially detectable (check if address resolves to a known mail forwarding service).

5. **Inconsistent return policy**: Different return windows stated in different places (e.g., "30 days" on one page, "14 days" on another). Detectable by crawling multiple policy pages.

6. **Missing hyperlinks for contact information**: Contact email/phone displayed as plain text rather than clickable links. Detectable by crawling.

7. **Zero reviews displayed**: No review widget on product pages is a red flag for Google. Detectable by crawling.

### Shopify Platform Constraints

- **Inventory quantity**: Shopify's `json` filter excludes `inventory_quantity` for stores created after December 5, 2017. Visible stock counters are the only quantity signal available without Admin API.
- **Policy URL wiring**: Some stores link generic pages from Online Store → Pages instead of canonical Settings → Policies URLs. Check both paths.
- **Geolocation/market apps**: Automatic currency conversion can conflict with GMC's requirement for consistent currency from feed through checkout. Test with varying locale headers.
- **Print-on-demand / dropshipping**: Higher scrutiny on availability, fulfillment, and originality. Flag thin supplier content.

### Theme Detection

Detect theme family from:
- `<meta name="theme-name" content="...">` (Online Store 2.0)
- CSS variable signatures (`--color-base-*` = Dawn, `--body-bg-secondary` = Shella)
- Class name patterns (`.product__title` = Dawn, `.product-single__title` = Impulse)

| Theme | Stable selectors |
|-------|-----------------|
| Dawn | `.product__title`, `.product__description`, `.product-form__submit`, `variant-selects`, `[data-selected-variant]` |
| Impulse | `.product-single__title`, `.product__price` |
| Shella | `.prd-block` wrappers; use Shopify Analytics JSON for price |
| Unknown | Fall back to `window.ShopifyAnalytics` for price, SKU, name |

## Issue Signatures (Normalized)

```
price_mismatch:dom_vs_jsonld
price_mismatch:dom_vs_checkout
fake_urgency:countdown_resets
fake_urgency:only_x_left_static
review_mismatch:dom_vs_jsonld
claim_unverified:{claim_type}
badge_unverified:{badge_name}
availability_conflict:button_vs_jsonld
schema:empty_field:{field_name}
schema:missing_gtin
stale_year:description_or_meta
policy_missing:{policy_type}
policy_error:{policy_url}:{status_code}
liquid_error:product_page
robots_block:googlebot
address_mismatch:footer_vs_contact
```

## Remediation Priority

Never submit a GMC appeal until Phase 1 and Phase 2 are resolved. Google limits review attempts and may impose cooldowns.

| Phase | Focus | Priority checks |
|-------|-------|----------------|
| 1 🔴 Do today | False claims & identity | Remove fake certifications; fix business identity; stop appealing until clean; fix Liquid errors |
| 2 🔴 Do this week | Purchase path & policies | Fix add-to-cart; add return address; define ambiguous warranties; align free-shipping claims; verify GMC account address matches website |
| 3 🟠 Do this month | Schema & product data | Populate empty schema fields; update stale content; remove unverifiable award claims; fix broken policy links |
| 4 🟡 Ongoing | Trust & freshness | Verify donation links; update year references; full product rescan; monitor feed vs page consistency |
