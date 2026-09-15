# Pencil Design Examples

Real-world examples of using Pencil MCP with Hermes Agent.

## Example 1: Create a Mobile App Login Screen

**User Request**: "Create a mobile app login screen with email and password fields, a login button, and a sign-up link."

**Agent Workflow**:

```
Step 1: Open new design document
  open_document(filePath="~/designs/login-app.pen")

Step 2: Create the main screen frame
  container=I("document",{
    type:"frame",
    name:"LoginScreen",
    width:375,           // iPhone width
    height:812,          // iPhone height
    fill:"#FFFFFF",
    layout:"vertical",
    alignItems:"center",
    justifyContent:"center",
    gap:24,
    padding:24
  })

Step 3: Add title
  title=I(container,{
    type:"text",
    content:"Welcome Back",
    fontSize:28,
    fontWeight:"bold",
    textAlign:"center"
  })

Step 4: Add subtitle
  subtitle=I(container,{
    type:"text",
    content:"Sign in to your account",
    fontSize:14,
    fill:"#666666"
  })

Step 5: Create form group (email + password)
  form=I(container,{
    type:"frame",
    layout:"vertical",
    gap:12,
    width:"100%"
  })

Step 6: Email input field
  email_input=I(form,{
    type:"frame",
    width:"100%",
    height:48,
    fill:"#F5F5F5",
    cornerRadius:8,
    padding:12,
    layout:"horizontal",
    alignItems:"center"
  })
  email_label=I(email_input,{
    type:"text",
    content:"📧",
    fontSize:16
  })
  email_text=I(email_input,{
    type:"text",
    content:"Email address",
    fontSize:14,
    fill:"#999999"
  })

Step 7: Password input field (similar structure)
  password_input=I(form,{
    type:"frame",
    width:"100%",
    height:48,
    fill:"#F5F5F5",
    cornerRadius:8,
    padding:12,
    layout:"horizontal",
    alignItems:"center"
  })
  password_icon=I(password_input,{
    type:"text",
    content:"🔒",
    fontSize:16
  })
  password_text=I(password_input,{
    type:"text",
    content:"Password",
    fontSize:14,
    fill:"#999999"
  })

Step 8: Create login button
  login_btn=I(container,{
    type:"frame",
    name:"LoginButton",
    width:"100%",
    height:48,
    fill:"#1E88E5",
    cornerRadius:8,
    layout:"horizontal",
    alignItems:"center",
    justifyContent:"center"
  })
  btn_text=I(login_btn,{
    type:"text",
    content:"Sign In",
    fontSize:16,
    fontWeight:"bold",
    fill:"#FFFFFF"
  })

Step 9: Add sign-up link
  footer=I(container,{
    type:"frame",
    layout:"horizontal",
    gap:4,
    justifyContent:"center"
  })
  footer_text=I(footer,{
    type:"text",
    content:"Don't have an account?",
    fontSize:12,
    fill:"#666666"
  })
  signup_link=I(footer,{
    type:"text",
    content:"Sign Up",
    fontSize:12,
    fill:"#1E88E5",
    textDecoration:"underline"
  })

Step 10: Verify design with screenshot
  screenshot=get_screenshot(filePath="~/designs/login-app.pen")
  → Returns image preview

Result: Beautiful mobile login screen ready for handoff to developers!
```

---

## Example 2: Update Brand Colors Across Design System

**User Request**: "Our brand color changed from blue (#0066CC) to purple (#7B2BE0). Update all components in the design system."

**Agent Workflow**:

```
Step 1: Get current design variables
  vars=get_variables(filePath="~/designs/design-system.pen")
  → Returns all defined color variables

Step 2: Find all components using the old blue color
  old_usage=search_all_unique_properties(
    filePath="~/designs/design-system.pen",
    property="fill"
  )
  → Finds ["#0066CC", "#FF5733", ...]

Step 3: Bulk replace the old color with new color
  result=replace_all_matching_properties(
    filePath="~/designs/design-system.pen",
    property="fill",
    oldValue="#0066CC",
    newValue="#7B2BE0"
  )
  → Returns {replaced: 47, affectedNodes: ["button-1", "card-1", ...]}

Step 4: Update design variable tokens
  set_variables(filePath="~/designs/design-system.pen",
    variables={
      color: {
        primary: "#7B2BE0",
        primaryLight: "#B4A7FF",
        primaryDark: "#5C1E7B"
      }
    }
  )

Step 5: Verify changes with screenshots
  screenshot=get_screenshot(filePath="~/designs/design-system.pen")
  → Shows all components with new purple color

Result: All 47 components updated in seconds, no manual editing needed!
```

---

## Example 3: Export All Screens as Marketing Assets

**User Request**: "Export all app screens as high-resolution PNG images for our website."

**Agent Workflow**:

```
Step 1: Fetch all screen nodes
  screens=batch_get(
    filePath="~/designs/app.pen",
    query="*Screen"  // matches HomeScreen, ProfileScreen, etc.
  )
  → Returns {
      HomeScreen: {id: "screen-1", ...},
      ProfileScreen: {id: "screen-2", ...},
      SettingsScreen: {id: "screen-3", ...}
    }

Step 2: Export each screen at 2x resolution
  exports=export_nodes(
    filePath="~/designs/app.pen",
    nodeIds=["screen-1", "screen-2", "screen-3"],
    format="png",
    scale=2  // 2x for Retina displays
  )
  → Returns {
      screen-1: "~/exports/HomeScreen@2x.png",
      screen-2: "~/exports/ProfileScreen@2x.png",
      screen-3: "~/exports/SettingsScreen@2x.png"
    }

Step 3: Create a summary
  User gets:
    ✅ HomeScreen@2x.png (750x1624px)
    ✅ ProfileScreen@2x.png (750x1624px)
    ✅ SettingsScreen@2x.png (750x1624px)

Result: Ready-to-upload marketing assets in minutes!
```

---

## Example 4: Analyze Design System Consistency

**User Request**: "Check if our design system has consistent spacing and typography."

**Agent Workflow**:

```
Step 1: Analyze layout and spacing
  layout=snapshot_layout(filePath="~/designs/design-system.pen")
  → Returns detailed spacing metrics

Step 2: Get all font sizes in use
  sizes=search_all_unique_properties(
    filePath="~/designs/design-system.pen",
    property="fontSize"
  )
  → Returns [12, 14, 14, 16, 16, 16, 18, 20, 24, ...]

Step 3: Get all spacing values
  spacing=search_all_unique_properties(
    filePath="~/designs/design-system.pen",
    property="gap"
  )
  → Returns [4, 8, 8, 8, 12, 12, 16, 16, 16, 24, ...]

Step 4: Analyze guidelines
  guidelines=get_guidelines(filePath="~/designs/design-system.pen")
  → Returns defined rules and standards

Step 5: Report to user
  Agent reports:
    ✅ Typography scale: [12, 14, 16, 18, 20, 24] (well-scaled)
    ✅ Spacing scale: [4, 8, 12, 16, 24] (good progression)
    ⚠️  14px used 4 times, consider consolidating
    ⚠️  Inconsistent padding in cards (12px vs 16px)

  Recommendation: Align padding to single 16px standard
```

---

## Example 5: Create Light/Dark Theme Variants

**User Request**: "Create a dark theme version of our app design."

**Agent Workflow**:

```
Step 1: Get current (light) theme variables
  light_vars=get_variables(filePath="~/designs/app.pen")
  → Returns:
    {
      color: {
        bg: "#FFFFFF",
        text: "#000000",
        border: "#E0E0E0"
      }
    }

Step 2: Define dark theme colors
  set_variables(
    filePath="~/designs/app.pen",
    variables={
      color: {
        bg: "#121212",
        text: "#FFFFFF",
        border: "#333333"
      }
    },
    theme="dark"
  )

Step 3: Take dark theme screenshot
  dark_screenshot=get_screenshot(filePath="~/designs/app.pen")
  → Shows app in dark colors

Step 4: Export both light and dark screens
  light=export_nodes(filePath="~/designs/app.pen", nodeIds=["HomeScreen"], scale=2)
  → ~/exports/HomeScreen-light@2x.png

  // Switch to dark theme
  set_variables(filePath="~/designs/app.pen", theme="dark")
  dark=export_nodes(filePath="~/designs/app.pen", nodeIds=["HomeScreen"], scale=2)
  → ~/exports/HomeScreen-dark@2x.png

Step 5: Deliver both variants
  User gets:
    ✅ HomeScreen-light@2x.png (light theme)
    ✅ HomeScreen-dark@2x.png (dark theme)

Result: Dual theme variants ready for development handoff!
```

---

## Example 6: Programmatic Component Library Setup

**User Request**: "Create a component library with Button, Card, and Input variants."

**Agent Workflow**:

```
Step 1: Open design system file
  open_document(filePath="~/designs/components.pen")

Step 2: Create Button component folder
  button_group=I("document",{
    type:"frame",
    name:"Button Components",
    layout:"horizontal",
    gap:24
  })

Step 3: Create Button.Primary variant
  btn_primary=I(button_group,{
    type:"frame",
    name:"Button.Primary",
    reusable:true,
    width:120,
    height:44,
    fill:"#1E88E5",
    cornerRadius:8,
    layout:"horizontal",
    alignItems:"center",
    justifyContent:"center",
    padding:12
  })
  btn_text=I(btn_primary,{
    type:"text",
    content:"Button",
    fontSize:14,
    fill:"#FFFFFF"
  })

Step 4: Create Button.Secondary variant
  btn_secondary=I(button_group,{
    type:"frame",
    name:"Button.Secondary",
    reusable:true,
    width:120,
    height:44,
    fill:"#FFFFFF",
    stroke:"#1E88E5",
    strokeWidth:2,
    cornerRadius:8,
    layout:"horizontal",
    alignItems:"center",
    justifyContent:"center",
    padding:12
  })
  btn_text=I(btn_secondary,{
    type:"text",
    content:"Button",
    fontSize:14,
    fill:"#1E88E5"
  })

Step 5: Create Card component
  card_group=I("document",{
    type:"frame",
    name:"Card Components",
    layout:"vertical",
    gap:24
  })
  
  card_default=I(card_group,{
    type:"frame",
    name:"Card.Default",
    reusable:true,
    width:300,
    height:200,
    fill:"#FFFFFF",
    stroke:"#E0E0E0",
    cornerRadius:8,
    padding:16,
    layout:"vertical",
    gap:12
  })

Step 6: Set up design tokens
  set_variables(filePath="~/designs/components.pen",
    variables={
      spacing: {
        xs: 4, sm: 8, md: 16, lg: 24, xl: 32
      },
      radius: {
        sm: 4, md: 8, lg: 16
      },
      color: {
        primary: "#1E88E5",
        secondary: "#43A047",
        error: "#D32F2F"
      }
    }
  )

Step 7: Export component library showcase
  export_nodes(filePath="~/designs/components.pen",
    nodeIds=["Button Components", "Card Components"],
    format="png"
  )

Result: Reusable component library set up and documented!
```

---

## Example 7: Convert Figma-Style Tokens to Hermes Variables

**User Request**: "I have design tokens from Figma. Update our Hermes design system with them."

**Agent Workflow**:

```
Step 1: Parse/import token data
  tokens_from_figma={
    "color.brand.primary": "#FF6B6B",
    "color.brand.secondary": "#4ECDC4",
    "typography.heading.large.size": 32,
    "typography.heading.large.weight": "bold",
    "spacing.unit": 8,
  }

Step 2: Map to Hermes variable structure
  set_variables(filePath="~/designs/system.pen",
    variables={
      color: {
        brand: {
          primary: "#FF6B6B",
          secondary: "#4ECDC4"
        }
      },
      typography: {
        heading: {
          large: {
            size: 32,
            weight: "bold"
          }
        }
      },
      spacing: {
        unit: 8,
        xs: 4,
        sm: 8,
        md: 16,
        lg: 24,
        xl: 32
      }
    }
  )

Step 3: Verify all components updated
  layout=snapshot_layout(filePath="~/designs/system.pen")
  → Confirms all variables applied

Result: Figma tokens integrated into Hermes design system!
```

---

## Tips for Agent Prompting

When asking Hermes to use Pencil MCP for design tasks:

**Good prompts:**
- "Create a mobile app login screen with..."
- "Update all buttons in the design system to use the new blue (#..."
- "Export all screens as PNG for the marketing team"
- "Analyze spacing consistency across components"
- "Create dark/light theme variants"

**Weak prompts:**
- "Make a nice design" (too vague)
- "Update the design" (missing details)
- "Fix it" (unclear what to fix)

**Enable agent success:**
- Specify file paths (absolute preferred)
- Mention component/element names
- Clarify output format (PNG, SVG, etc.)
- Include color codes or token names
- Mention screen sizes (mobile=375px width, tablet=600px, etc.)
