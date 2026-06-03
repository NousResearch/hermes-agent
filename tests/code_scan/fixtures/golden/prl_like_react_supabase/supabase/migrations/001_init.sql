-- Tiny synthetic Supabase migration for UA fixture
CREATE TABLE tasks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title text NOT NULL,
  status text DEFAULT 'todo',
  created_at timestamptz DEFAULT now()
);

ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "tasks_select_anon" ON tasks
  FOR SELECT
  TO anon
  USING (true);
