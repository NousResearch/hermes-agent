import React from 'react';
import { createClient } from '@supabase/supabase-js';
import ManagerDesktopNotice from './components/ManagerDesktopNotice';
import { summarizeTaskStatus } from './api/tasks';

const supabase = createClient(
  'https://example.supabase.co',
  'public-anon-key-fixture'
);

function App({ sessionSummary }) {
  const taskCounts = summarizeTaskStatus([
    { id: 1, title: 'Synthetic worksite check', status: 'open' },
  ]);

  return (
    <div>
      <h1>PRL-like Fixture</h1>
      <ManagerDesktopNotice />
      <p>Role: {sessionSummary?.role ?? 'anonymous'}</p>
      <p>Open tasks: {taskCounts.open ?? 0}</p>
    </div>
  );
}

export { supabase };
export default App;
