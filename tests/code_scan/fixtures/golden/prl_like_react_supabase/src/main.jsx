import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { getSessionSummary } from './auth/session';

const rootElement = document.getElementById('root');

if (rootElement) {
  createRoot(rootElement).render(
    <React.StrictMode>
      <App sessionSummary={getSessionSummary()} />
    </React.StrictMode>
  );
}
