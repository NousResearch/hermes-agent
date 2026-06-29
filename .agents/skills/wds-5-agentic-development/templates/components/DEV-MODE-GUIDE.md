# Dev Mode - Usage Guide

**Purpose**: Easy feedback on prototypes by copying Object IDs to clipboard

---

## 🎯 What is Dev Mode?

Dev Mode is a built-in feature in all WDS prototypes that allows testers, stakeholders, and designers to easily reference specific UI elements when providing feedback.

Instead of saying *"The button in the top right"*, you can say *"Fix `customer-sign-bankid`"* - precise and unambiguous!

---

## 🚀 How to Use

### Step 1: Activate Dev Mode

**Two ways**:
1. Click the **Dev Mode button** (top-right corner)
2. Press **Ctrl+E** on your keyboard

The button will turn blue and say **"Dev Mode: ON"**

---

### Step 2: Find the Element

- **Hover** over any element you want to reference
- You'll see a **gray outline** appear
- A **tooltip** shows the Object ID

**Prototype still works normally!** You can click buttons, fill forms, etc.

---

### Step 3: Copy the Object ID

- **Hold the Shift key** (outline turns **green**)
- **Click the element** while holding Shift
- **Object ID is copied!** ✓

You'll see a green success message: **"✓ Copied: [object-id]"**

**Important**: Shift key is **disabled when typing in form fields** (input, textarea, etc.) so you can type capital letters and special characters normally!

---

### Step 4: Paste in Feedback

Now paste the Object ID in your feedback:

**Good feedback**:
```
❌ Issue with `customer-sign-bankid`:
The button is disabled even after I check the consent checkbox.

💡 Suggestion for `sidebar-video`:
Make the video auto-play on mobile.
```

**Developer knows EXACTLY** which element you're talking about!

---

## 🎨 Visual Guide

| State | Appearance | Action |
|-------|------------|--------|
| **Dev Mode OFF** | Normal prototype | Click button or press Ctrl+E |
| **Dev Mode ON (hovering)** | Gray outline | Shows Object ID in tooltip |
| **Shift held (hovering)** | Green outline | Click to copy |
| **After copying** | Green flash | Object ID in clipboard |

---

## ⌨️ Keyboard Shortcuts

- **Ctrl+E**: Toggle Dev Mode on/off
- **Shift + Click**: Copy Object ID (when dev mode is on)

---

## 💡 Tips

1. **Activate once**, then navigate through prototype normally
2. **Hold Shift only when copying** - prototype works without it
3. **Type in fields normally** - Shift is disabled when focused on input/textarea
4. **Deactivate when done** testing (Ctrl+E again)
5. **Object IDs are permanent** - always refer to the same element

---

## 📋 Example Workflow

### Tester's Perspective:

1. Open prototype
2. Press **Ctrl+E** (Dev Mode on)
3. Test the prototype normally
4. Find a bug - hover over problem element
5. Hold **Shift**, click element
6. Paste Object ID into bug report: "`customer-facility-startdate-group` shows wrong default date"
7. Continue testing

### Designer's Perspective:

Receives feedback:
```
Bug: `customer-facility-startdate-group` shows wrong default date
```

- Open prototype
- Press **Ctrl+F** in browser, search for `customer-facility-startdate-group`
- Find exact element in code
- Fix the date calculation
- Done! ✅

---

## 🔧 For Developers

When you receive Object IDs in feedback:

1. Open the HTML file
2. Search for the Object ID (Ctrl+F)
3. Element is either:
   - `id="object-id"` attribute
   - `data-object-id="object-id"` attribute
4. Fix the issue in that specific element

---

## ❓ FAQs

**Q: Does Dev Mode affect the prototype?**  
A: No! The prototype works normally. You need to hold Shift to copy IDs.

**Q: Can I use this on mobile?**  
A: Yes! The button appears on mobile too. Use a Bluetooth keyboard or on-screen Shift key.

**Q: Can I type in form fields while Dev Mode is on?**  
A: Yes! Shift key is automatically disabled when you're typing in input fields or textareas, so you can type capital letters and special characters normally.

**Q: What if an element doesn't have an ID?**  
A: Dev Mode walks up the tree to find the nearest parent with an ID.

**Q: Can I copy multiple IDs?**  
A: Yes! Hold Shift, click first element, release Shift, hold again, click second element, etc.

**Q: Is this only for bugs?**  
A: No! Use it for any feedback - bugs, suggestions, questions, clarifications.

---

## 🎓 Best Practices

### For Testers:
- ✅ **DO**: Include Object ID in every piece of feedback
- ✅ **DO**: Test prototype normally, copy IDs when needed
- ✅ **DO**: Combine Object ID with description
- ❌ **DON'T**: Leave Dev Mode on during normal use

### For Designers:
- ✅ **DO**: Ensure all interactive elements have Object IDs
- ✅ **DO**: Use descriptive, consistent naming
- ✅ **DO**: Include Dev Mode in all prototypes
- ❌ **DON'T**: Change Object IDs after sharing prototype

---

## 🚨 Troubleshooting

**Problem**: Dev Mode button not showing  
**Solution**: Check that `dev-mode.js` and `dev-mode.css` are loaded

**Problem**: Clicking doesn't copy  
**Solution**: Make sure you're holding **Shift** while clicking

**Problem**: Tooltip not showing  
**Solution**: Element might not have an ID - check console logs

**Problem**: Can't turn off Dev Mode  
**Solution**: Press Ctrl+E or refresh the page

---

**Dev Mode makes feedback precise, fast, and frustration-free!** 🎯

