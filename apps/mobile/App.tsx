import { QueryClientProvider } from '@tanstack/react-query';
import { StatusBar } from 'expo-status-bar';
import { StrictMode } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import DesktopApp from './src/hermes-app';
import { ErrorBoundary } from './src/components/error-boundary';
import { HapticsProvider } from './src/components/haptics-provider';
import { I18nProvider } from './src/i18n';
import { queryClient } from './src/lib/query-client';
import { ThemeProvider } from './src/themes/context';

// Mobile port entrypoint for the copied Hermes Desktop renderer.
// This deliberately imports the real Desktop app tree from ./src/app.
// Compatibility work should happen as small shims/patches around copied files,
// not by replacing the app with a bespoke React Native mock.
export default function App() {
  return (
    <StrictMode>
      <SafeAreaProvider>
        <ErrorBoundary label="mobile-root">
          <QueryClientProvider client={queryClient}>
            <I18nProvider>
              <ThemeProvider>
                <HapticsProvider>
                  <MemoryRouter initialEntries={['/']}>
                    <DesktopApp />
                  </MemoryRouter>
                </HapticsProvider>
              </ThemeProvider>
            </I18nProvider>
          </QueryClientProvider>
        </ErrorBoundary>
        <StatusBar style="dark" />
      </SafeAreaProvider>
    </StrictMode>
  );
}
