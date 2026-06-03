import { createClient } from '@supabase/supabase-js';

// Minimal synthetic client stub for fixture shape (no real keys)
export const supabase = createClient(
  process.env.SUPABASE_URL || 'https://example.supabase.co',
  process.env.SUPABASE_ANON || 'fixture-anon-key'
);

export default supabase;
