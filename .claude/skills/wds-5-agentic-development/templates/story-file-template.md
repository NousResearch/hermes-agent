# Story [Page].[Section]: [Page Name] - [Section Name]

**Page**: [Page Number] [Page Name]  
**Section**: [N] of [Total]  
**Complexity**: Simple | Medium | Complex  
**Estimated Time**: [X] minutes

---

## 🎯 Goal

[Brief description of what this section accomplishes]

---

## 📋 What to Build

### HTML Elements

```html
<!-- [Description of HTML to add] -->
<div class="[tailwind-classes]">
    <!-- Specific HTML structure here -->
</div>
```

### JavaScript (if needed)

```javascript
// [Description of JavaScript functionality]
function [functionName]() {
    // Implementation
}
```

### Tailwind Classes to Use

**Key classes for this section**:
- `[class-category]`: `[specific-classes]`
- `[class-category]`: `[specific-classes]`

**Example combinations**:
```html
<!-- Input field -->
<input class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[project]-500">

<!-- Button -->
<button class="w-full py-3 bg-[project]-600 text-white rounded-lg font-semibold hover:bg-[project]-700 transition-colors">
```

---

## 🔗 Dependencies

**Shared code**:
- ✅ `shared/prototype-api.js` (already loaded)
- ✅ `shared/init.js` (already loaded)

**Components** (load if not already included):
- [ ] `components/image-crop.js` (if using image upload)
- [ ] `components/toast.js` (if showing notifications)
- [ ] `components/modal.js` (if using modals)
- [ ] `components/form-validation.js` (if validating forms)

---

## 📸 Baseline State

_Capture with Puppeteer before implementation when modifying existing features. Skip for new sections._

| Element | Current Value | Notes |
|---------|---------------|-------|
| [element] | [current value] | [any relevant context] |

---

## 📝 Implementation Steps

### Step 1: [First Step]
[What to do first]

### Step 2: [Second Step]
[What to do second]

### Step 3: [Third Step]
[What to do third]

---

## ✅ Acceptance Criteria

### Agent-Verifiable (Puppeteer)

| # | Criterion | Element | Expected | How to Verify |
|---|-----------|---------|----------|---------------|
| 1 | [Criterion] | `[selector]` | [Value] | [Method] |
| 2 | [Criterion] | `[selector]` | [Value] | [Method] |
| 3 | [Criterion] | `[selector]` | [Value] | [Method] |

### User-Evaluable (Qualitative)

- [ ] Flow feels natural and intuitive
- [ ] Visual hierarchy guides the eye correctly
- [ ] Section feels consistent with the rest of the prototype
- [ ] [Additional qualitative criterion]

---

## 🧪 How to Test

### Puppeteer Self-Verification (Agent)

Before presenting to user:

1. Open `[Page-Number]-[Page-Name].html` in Puppeteer
2. Set viewport to target width (375px for mobile)
3. For each agent-verifiable criterion in the table above:
   - Locate element
   - Read actual value
   - Compare to expected
   - Narrate with ✓/✗
4. Fix any mismatches and re-verify until all pass
5. Check console for errors

See [Inline Testing Guide](../guides/INLINE-TESTING-GUIDE.md) for full methodology.

### User Qualitative Review

After Puppeteer verification passes, present to user:
- Summarize Puppeteer results (X/Y criteria pass)
- Ask user to evaluate qualitative criteria above
- Collect feedback on feel, flow, clarity, consistency

---

## 🐛 Common Issues & Fixes

### Issue: [Problem Description]
**Symptom**: [What user sees]  
**Cause**: [Why it happens]  
**Fix**: [How to fix it]

### Issue: [Problem Description]
**Symptom**: [What user sees]  
**Cause**: [Why it happens]  
**Fix**: [How to fix it]

---

## 🎨 Design Notes

**Visual requirements**:
- [Design consideration 1]
- [Design consideration 2]

**UX considerations**:
- [UX note 1]
- [UX note 2]

---

## 💡 Tips

- [Helpful tip 1]
- [Helpful tip 2]

---

## ➡️ Next Section

After this section is approved: `[Page].[NextSection]-[page-name]-[next-section-name].md`

---

## 📊 Status Tracking

**Status**: ⏸️ Not Started | 🚧 In Progress | ✅ Complete  
**Started**: [Date/Time]  
**Completed**: [Date/Time]  
**Approved By**: [Name]  
**Notes**: [Any special notes or changes made]

---

## 🔄 Changes from Original Plan

*Document any deviations from the work file plan here:*

- [Change 1 and reason]
- [Change 2 and reason]

