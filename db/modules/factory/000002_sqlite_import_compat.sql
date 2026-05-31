-- Compatibility columns for importing the existing local Factory SQLite progress DB.
ALTER TABLE factory.agents ADD COLUMN IF NOT EXISTS display_name text;
ALTER TABLE factory.agents ADD COLUMN IF NOT EXISTS preferred_engine text;

ALTER TABLE factory.tasks ADD COLUMN IF NOT EXISTS reviewer_profile text;
ALTER TABLE factory.tasks ADD COLUMN IF NOT EXISTS evidence_required boolean NOT NULL DEFAULT true;
ALTER TABLE factory.tasks ADD COLUMN IF NOT EXISTS evidence_status text NOT NULL DEFAULT 'missing';
ALTER TABLE factory.tasks ADD COLUMN IF NOT EXISTS risk_level text NOT NULL DEFAULT 'medium';

ALTER TABLE factory.events ADD COLUMN IF NOT EXISTS lane_id text REFERENCES factory.lanes(lane_id) ON DELETE SET NULL;
ALTER TABLE factory.gates ADD COLUMN IF NOT EXISTS lane_id text REFERENCES factory.lanes(lane_id) ON DELETE SET NULL;
ALTER TABLE factory.artifacts ADD COLUMN IF NOT EXISTS lane_id text REFERENCES factory.lanes(lane_id) ON DELETE SET NULL;
