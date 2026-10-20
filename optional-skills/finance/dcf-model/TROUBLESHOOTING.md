# DCF Model Troubleshooting Guide

**When to read this file:** If recalc.py shows errors OR valuation results seem unreasonable OR case selector not working properly.

## Model Returns Error Values

### #REF! Errors
- Usually caused by formulas referencing wrong rows after headers were inserted
- Solution: Rebuild with correct row references, or start over following layout planning
- Prevention: Define all row positions BEFORE writing formulas

### #DIV/0! Errors
- Division by zero or empty cells
- Solution: Add IF statements to handle zeros: `=IF([Divisor]=0,0,[Numerator]/[Divisor])`

### #VALUE! Errors
- Wrong data type in calculation (text instead of number)
- Solution: Verify all inputs are formatted as numbers

## Valuation Seems Unreasonable

### Implied price far too high
- Check terminal value isn't >80% of EV
- Verify terminal growth < WACC
- Review if growth assumptions are realistic
- Consider if margins are too optimistic

### Implied price far too low
- Verify net debt vs net cash is correct
- Check if WACC is too high
- Review if projections are too conservative
- Consider if terminal growth is too low

## Case Selector Not Working

### Consolidation column not updating when switching scenarios
- Verify case selector cell contains 1, 2, or 3
- Check INDEX/OFFSET formulas reference correct row range and selector cell
- Ensure absolute references ($B$6) are used for selector
- Test by manually changing the selector cell and verifying projection values update
