import React from 'react';
import { createClient } from '@supabase/supabase-js';
import ManagerDesktopNotice from './components/ManagerDesktopNotice';
import { fetchTasks } from '@lib/api';

const supabase = createClient(
  'https://example.supabase.co',
  'public-anon-key-fixture'
);

function App() {
  return (
    <div>
      <h1>PRL-like Fixture</h1>
      <ManagerDesktopNotice />
      <p>Tasks: {fetchTasks().length}</p>
    </div>
  );
}

export default App;
