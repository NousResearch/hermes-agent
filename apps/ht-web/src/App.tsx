import { lazy, Suspense } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";
import { GatewayProvider } from "./gateway/GatewayContext";
import { AppShell } from "./app/AppShell";
import ChatPage from "./pages/ChatPage";

// Management pages are lazy-loaded so the chat bundle stays lean; they pull in
// the REST client and their own widgets only when first navigated to.
const SessionsPage = lazy(() => import("./pages/SessionsPage"));
const ModelsPage = lazy(() => import("./pages/ModelsPage"));
const ConfigPage = lazy(() => import("./pages/ConfigPage"));
const LogsPage = lazy(() => import("./pages/LogsPage"));
const SystemPage = lazy(() => import("./pages/SystemPage"));

function Loading() {
  return <p className="ht-muted" style={{ padding: 24 }}>Loading…</p>;
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
            <Route
              path="sessions"
              element={
                <Suspense fallback={<Loading />}>
                  <SessionsPage />
                </Suspense>
              }
            />
            <Route
              path="models"
              element={
                <Suspense fallback={<Loading />}>
                  <ModelsPage />
                </Suspense>
              }
            />
            <Route
              path="config"
              element={
                <Suspense fallback={<Loading />}>
                  <ConfigPage />
                </Suspense>
              }
            />
            <Route
              path="logs"
              element={
                <Suspense fallback={<Loading />}>
                  <LogsPage />
                </Suspense>
              }
            />
            <Route
              path="system"
              element={
                <Suspense fallback={<Loading />}>
                  <SystemPage />
                </Suspense>
              }
            />
          </Route>
        </Routes>
      </HashRouter>
    </GatewayProvider>
  );
}
