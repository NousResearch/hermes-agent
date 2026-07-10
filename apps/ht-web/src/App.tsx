import { lazy, Suspense, type ComponentType } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";
import { GatewayProvider } from "./gateway/GatewayContext";
import { AppShell } from "./app/AppShell";
import ChatPage from "./pages/ChatPage";

// Management pages are lazy-loaded so the chat bundle stays lean; each pulls in
// the REST client and its own widgets only when first navigated to. Keyed by
// route path — mirror src/app/nav.ts so nav and routes stay in lockstep.
const MANAGEMENT_ROUTES: Record<string, ComponentType> = {
  sessions: lazy(() => import("./pages/SessionsPage")),
  models: lazy(() => import("./pages/ModelsPage")),
  skills: lazy(() => import("./pages/SkillsPage")),
  plugins: lazy(() => import("./pages/PluginsPage")),
  mcp: lazy(() => import("./pages/McpPage")),
  cron: lazy(() => import("./pages/CronPage")),
  channels: lazy(() => import("./pages/ChannelsPage")),
  webhooks: lazy(() => import("./pages/WebhooksPage")),
  pairing: lazy(() => import("./pages/PairingPage")),
  profiles: lazy(() => import("./pages/ProfilesPage")),
  files: lazy(() => import("./pages/FilesPage")),
  config: lazy(() => import("./pages/ConfigPage")),
  env: lazy(() => import("./pages/EnvPage")),
  analytics: lazy(() => import("./pages/AnalyticsPage")),
  logs: lazy(() => import("./pages/LogsPage")),
  system: lazy(() => import("./pages/SystemPage")),
};

function Loading() {
  return (
    <p className="ht-muted" style={{ padding: 24 }}>
      Loading…
    </p>
  );
}

export default function App() {
  return (
    <GatewayProvider>
      {/* Hash routing: the SPA is served as a static bundle by the Python
          gateway with no server-side route table, so hash routes avoid 404s
          on deep-link refresh without needing a catch-all rewrite. */}
      <HashRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<ChatPage />} />
            {Object.entries(MANAGEMENT_ROUTES).map(([path, Page]) => (
              <Route
                key={path}
                path={path}
                element={
                  <Suspense fallback={<Loading />}>
                    <Page />
                  </Suspense>
                }
              />
            ))}
          </Route>
        </Routes>
      </HashRouter>
    </GatewayProvider>
  );
}
