# Scenario [Number]: [Scenario Name] - Prototype Roadmap

**Scenario**: [Scenario Name]  
**Pages**: [First] through [Last]  
**Device Compatibility**: [Type] ([Width range])  
**Last Updated**: [Date]

---

## 🎯 Scenario Overview

**User Journey**: [Brief description of complete user flow]

**Pages in this Scenario**:
1. [Page 1] - [Description]
2. [Page 2] - [Description]
3. [Page 3] - [Description]
...

---

## 📱 Device Compatibility

**Type**: [Mobile-Only | Mobile + Tablet | Fully Responsive | Desktop-Only]

**Reasoning**: 
[Why this device compatibility was chosen for this scenario]

**Test Viewports**:
- [Device 1] ([width]px × [height]px) - [Purpose]
- [Device 2] ([width]px × [height]px) - [Purpose]
- [Device 3] ([width]px × [height]px) - [Purpose]

**Optimization Strategy**:
- ✅ [Optimization 1]
- ✅ [Optimization 2]
- ✅ [Optimization 3]
- ❌ [Not included 1]
- ❌ [Not included 2]

**Tailwind Approach**:
```html
<!-- [Brief description of Tailwind strategy] -->
```

---

## 📁 Folder Structure

**HTML Files** (root level - double-click to open):
```
[Page-1].html
[Page-2].html
[Page-3].html
...
```

**Supporting Folders**:
- `shared/` - Shared code (ONE COPY for all pages)
- `components/` - Reusable UI components (ONE COPY)
- `pages/` - Page-specific scripts (only for complex pages)
- `data/` - Demo data (auto-loads on first use)
- `stories/` - Section development documentation
- `work/` - Planning files (work.yaml for each page)

---

## 🚀 Quick Start

### For Testing
1. **Open** `[First-Page].html` (double-click)
2. **Demo data prompt** → Click YES
3. **Navigate** through the flow
4. **Data persists** across pages (sessionStorage)

### For Stakeholders
1. **Unzip** the Prototype folder
2. **Open** `[First-Page].html`
3. **Test** complete user journey
4. **Share feedback**

### For Developers
1. **Review** `work/` folder for specifications
2. **Check** `stories/` folder for implementation details
3. **Examine** `shared/prototype-api.js` for data operations
4. **Extract** HTML/Tailwind structure
5. **Migrate** to production (see TODOs in code)

---

## 🎨 Shared Resources (No Duplication!)

### `shared/prototype-api.js`
**Used by**: ALL prototypes  
**Purpose**: API abstraction layer (simulates backend with sessionStorage)

**Key methods**:
```javascript
PrototypeAPI.getUser()
PrototypeAPI.createUserProfile(userData)
PrototypeAPI.createFamily(familyData)
PrototypeAPI.addDog(dogData)
// ... see file for complete API
```

**Console commands** (for debugging):
```javascript
PrototypeAPI.getDebugInfo()  // See current state
PrototypeAPI.clearAllData()  // Reset everything
```

---

### `shared/init.js`
**Used by**: ALL prototypes  
**Purpose**: Auto-initialization (loads demo data, sets up page)

**What it does**:
- Checks if demo data exists
- Loads from `data/demo-data.json` if empty
- Calls `window.initPage()` if defined
- Logs current state to console

---

### `shared/utils.js`
**Used by**: ALL prototypes  
**Purpose**: Helper functions (date formatting, validation, etc.)

---

## 🧩 Components (Reusable - ONE COPY)

### `components/image-crop.js`
**Used by**: [Pages that use image upload]  
**Purpose**: Image upload with circular crop

**Usage**:
```javascript
ImageCrop.cropImage(file, { aspectRatio: 1 });
```

---

### `components/toast.js`
**Used by**: [Pages with notifications]  
**Purpose**: Success/error toast notifications

**Usage**:
```javascript
showToast('Success message!', 'success');
showToast('Error message', 'error');
```

---

### `components/modal.js`
**Used by**: [Pages with modals]  
**Purpose**: Generic modal overlay

---

### `components/form-validation.js`
**Used by**: [Pages with forms]  
**Purpose**: Real-time form validation

---

## 📊 Demo Data

### `data/demo-data.json`
**Purpose**: Complete demo dataset for scenario

**Contents**:
- User profile
- Family data
- [Other data entities]

**Edit this file** to change demo data (JSON format, designer-friendly)

---

### `data/[additional-data].json`
**Purpose**: [Description]

---

## 📋 Prototype Status

| Page | Status | Sections | Last Updated | Notes |
|------|--------|----------|--------------|-------|
| [Page 1] | ✅ Complete | 3/3 | [Date] | - |
| [Page 2] | ✅ Complete | 4/4 | [Date] | - |
| [Page 3] | 🚧 In Progress | 2/5 | [Date] | Building form fields |
| [Page 4] | ⏸️ Not Started | 0/6 | - | Planned |

**Status Legend**:
- ✅ Complete - All sections done, tested, approved
- 🚧 In Progress - Currently building section-by-section
- ⏸️ Not Started - Planned, not yet started
- 🔴 Blocked - Waiting on dependency

---

## 🔄 Development Workflow

### 1. Planning Phase
- Create work file: `work/[Page]-Work.yaml`
- Define sections (4-8 per page)
- Identify Object IDs
- List demo data needs
- Get approval

### 2. Implementation Phase
- Build section-by-section
- Create story files just-in-time
- Test after each section
- Get approval before next section
- File lives in root from start (no temp folder)

### 3. Finalization Phase
- Complete integration test
- Update status to Complete
- Document any changes
- Update this roadmap

---

## 🧪 Testing Requirements

### Functional Testing (All Pages)
- [ ] All form fields work
- [ ] Validation shows errors correctly
- [ ] Submit buttons work with loading states
- [ ] Success/error feedback displays
- [ ] Navigation works (back, next, cancel)
- [ ] Data persists across pages

### Device Testing
- [ ] [Primary viewport] ([width]px)
- [ ] [Secondary viewport] ([width]px)
- [ ] [Tertiary viewport] ([width]px)
- [ ] Portrait orientation
- [ ] Touch interactions work
- [ ] No horizontal scroll

### Browser Testing
- [ ] Chrome (primary)
- [ ] Safari (iOS/Mac)
- [ ] Firefox
- [ ] Edge

---

## 🎓 Tailwind Reference

### Project Colors
```javascript
// Tailwind config (in each HTML file)
'[project-name]': {
  50: '#eff6ff',
  500: '#2563eb',
  600: '#1d4ed8',
  700: '#1e40af',
}
```

**Usage**: `bg-[project-name]-600`, `text-[project-name]-500`, etc.

### Common Patterns

**Form Input**:
```html
<input class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-[project]-500">
```

**Primary Button**:
```html
<button class="w-full py-3 bg-[project]-600 text-white rounded-lg font-semibold hover:bg-[project]-700 transition-colors">
```

**Toast Notification**:
```html
<div class="fixed bottom-6 left-1/2 -translate-x-1/2 bg-gray-900 text-white px-6 py-3 rounded-lg shadow-lg">
```

---

## 🐛 Troubleshooting

### Issue: Demo data not loading
**Solution**: Check `data/demo-data.json` exists, check console for errors

### Issue: Tailwind not working
**Solution**: Check `<script src="https://cdn.tailwindcss.com"></script>` in `<head>`

### Issue: Navigation not working
**Solution**: Check relative paths (should be `[Page].html` from root)

### Issue: Shared code not loading
**Solution**: Check paths are `shared/[file].js`, `components/[file].js`

### Issue: Form not submitting
**Solution**: Check `event.preventDefault()` in `handleSubmit(event)`

---

## 📚 Documentation

**Work Files** (`work/` folder):
- High-level plans for each page
- Section breakdowns
- Object ID maps
- Acceptance criteria

**Story Files** (`stories/` folder):
- Detailed section implementation guides
- Created just-in-time during development
- Document what was actually built
- Include changes from original plan

---

## 🚀 Production Migration

### Steps to Production
1. **Replace** `PrototypeAPI` calls with real backend
2. **Migrate** sessionStorage to database
3. **Add** authentication layer
4. **Implement** proper error handling
5. **Add** loading states for real network delays
6. **Setup** Tailwind build process (vs CDN)
7. **Optimize** images and assets
8. **Test** with real data

### Migration Helpers
- Search for `TODO:` comments in code
- Check `PrototypeAPI` methods for Supabase equivalents
- Review work files for production requirements

---

## 📧 Support & Questions

**For design questions**: Review story files in `stories/` folder  
**For functionality questions**: Review work files in `work/` folder  
**For implementation details**: Check inline comments in HTML files  
**For API questions**: Review `shared/prototype-api.js` documentation

---

## 📊 Scenario Statistics

**Total Pages**: [N]  
**Completed**: [N]  
**In Progress**: [N]  
**Total Sections**: [N]  
**Object IDs**: [N]  
**Shared Components**: [N]  
**Demo Data Files**: [N]

**Estimated Test Time**: [X] minutes (complete flow)  
**Estimated Build Time**: [X] hours (all pages)

---

## 📝 Change Log

### [Date]
- [Change description]
- [Page affected]

### [Date]
- [Change description]
- [Page affected]

---

**Last Updated**: [Date]  
**Version**: 1.0  
**Status**: [In Development | Testing | Complete]

