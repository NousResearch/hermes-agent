export interface FixtureSessionSummary {
  role: string;
  authProvider: string;
  mfaRequired: boolean;
}

export function getSessionSummary(): FixtureSessionSummary {
  return {
    role: 'fixture-manager',
    authProvider: 'synthetic-supabase',
    mfaRequired: true,
  };
}

export function canAccessManagerDesktop(sessionSummary: FixtureSessionSummary): boolean {
  return sessionSummary.role === 'fixture-manager';
}
