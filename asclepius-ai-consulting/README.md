# ASCLEPIUS AI Consulting

`PROJECT_NAME = ASCLEPIUS AI Consulting`

**One-line pitch:** ASCLEPIUS sells “The AI Prescription”: a doctor-style diagnostic and implementation service for owner-operated small businesses that want practical AI wins without becoming AI experts.

## The four-tier model

1. **The Diagnostic — $999**: a 30–45 minute structured discovery call and written AI Prescription identifying the 3–5 highest-value bottlenecks, recommended tools/workflows, conservative savings, and implementation order.
2. **The Build — $2,500–$10,000+**: fixed-fee implementation of approved prescription items: setup, integrations, prompts, SOPs, and training.
3. **The Retainer — $500–$2,000/month**: ongoing AI ops, quarterly re-diagnosis, workflow tuning, tool evaluation, and staff onboarding.
4. **Office Hours / Overflow — $150–$300/hour**: ad hoc expert help for clients who need occasional hands-on support but are not ready for a retainer.

## Quickstart

```bash
# Run all tests
python -m unittest discover -s tests

# Create and inspect a local CRM database
python crm/crm.py --db crm/asclepius.db add-lead --name "Jordan Lee" --business "Lee HVAC" --source "warm_referral"
python crm/crm.py --db crm/asclepius.db list-leads
python crm/crm.py --db crm/asclepius.db update-status --lead-id 1 --status qualified
python crm/crm.py --db crm/asclepius.db add-engagement --lead-id 1 --tier Diagnostic --price 999 --status closed_won
python crm/crm.py --db crm/asclepius.db revenue-summary

# Pricing and ROI calculators
python finance/pricing-calculator.py --tier build --hours 20 --complexity 1.5
python finance/roi-calculator-for-prospects.py --hours-per-week 6 --hourly-value 150

# Open the landing page
python -m http.server 8000 --directory website
# then visit http://localhost:8000
```

## Design note

This is a standalone consulting venture, intentionally decoupled from HermesOS/RichieRichOS. The CRM schema is deliberately plain SQLite so it could later export into HermesOS if Richie chooses, but no integration is built now. That remains a future business/technical decision.
