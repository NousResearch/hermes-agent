# Key Page Dependency Trees

## /chat
Entry: `web/src/pages/ChatPage.tsx`
Dependencies:
- web/src/components/ChatSidebar.tsx
  - web/src/components/ChatSessionList.tsx
  - web/src/components/ModelPickerDialog.tsx
  - web/src/components/ReasoningPicker.tsx
- web/src/lib/api.ts
- web/src/lib/gatewayClient.ts
- web/src/lib/pty-reconnect.ts
- web/src/lib/pty-scroll.ts
- web/src/themes/context.tsx
- web/src/App.tsx (persistent shell host)

## /sessions
Entry: `web/src/pages/SessionsPage.tsx`
Dependencies:
- web/src/components/DeleteConfirmDialog.tsx
- web/src/components/Markdown.tsx
- web/src/components/ProfileScopeBanner.tsx
- web/src/lib/api.ts
- web/src/lib/session-refresh.ts
- web/src/App.tsx

## /models
Entry: `web/src/pages/ModelsPage.tsx`
Dependencies:
- web/src/components/ModelInfoCard.tsx
- web/src/components/ModelPickerDialog.tsx
- web/src/components/ModelReloadConfirm.tsx
- web/src/lib/api.ts
- web/src/App.tsx

## /config
Entry: `web/src/pages/ConfigPage.tsx`
Dependencies:
- web/src/components/AutoField.tsx
- web/src/components/ProfileScopeBanner.tsx
- web/src/lib/api.ts
- web/src/lib/nested.ts
- web/src/App.tsx

## Shared shell dependencies
- web/src/components/ProfileSwitcher.tsx
- web/src/components/ThemeSwitcher.tsx
- web/src/components/SidebarStatusStrip.tsx
- web/src/components/SidebarFooter.tsx
- web/src/index.css
- web/src/themes/context.tsx
- web/src/themes/presets.ts
- web/src/themes/types.ts

