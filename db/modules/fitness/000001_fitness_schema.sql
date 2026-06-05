-- Fitness Coach Core: personal profiles, goals, food/nutrition, exercises, routines, workouts, metrics, and coaching loops.
CREATE SCHEMA IF NOT EXISTS fitness;

INSERT INTO agent_core.modules(module, description, owner, schema_name)
VALUES ('fitness', 'Agent-native fitness coach core: profiles, goals, food cache, nutrition logs, exercise database, routines, workouts, body metrics, check-ins, and coaching recommendations.', 'agent-runtime', 'fitness')
ON CONFLICT (module) DO UPDATE SET updated_at = now();

INSERT INTO agent_core.module_databases(module, database_name, connection_role, migration_role, metadata)
VALUES ('fitness', current_database(), 'fitness_runtime', 'agent_admin', '{"option":"same-agent-db-schema","scope":"personal-fitness-coach-core","primary_sources":["Open Food Facts API/cache","free-exercise-db-compatible schema","custom user data"]}'::jsonb)
ON CONFLICT (module) DO UPDATE SET database_name = EXCLUDED.database_name, connection_role = EXCLUDED.connection_role, migration_role = EXCLUDED.migration_role, metadata = EXCLUDED.metadata;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'fitness_runtime') THEN
    CREATE ROLE fitness_runtime NOLOGIN;
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS fitness.profiles (
  profile_id text PRIMARY KEY,
  owner_id text,
  display_name text NOT NULL,
  sex text,
  birth_date date,
  height_cm numeric,
  activity_level text NOT NULL DEFAULT 'moderate',
  timezone text NOT NULL DEFAULT 'America/Caracas',
  preferred_units jsonb NOT NULL DEFAULT '{"weight":"kg","distance":"km"}'::jsonb,
  dietary_preferences jsonb NOT NULL DEFAULT '[]'::jsonb,
  allergies jsonb NOT NULL DEFAULT '[]'::jsonb,
  equipment_available jsonb NOT NULL DEFAULT '[]'::jsonb,
  injury_notes text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fitness.goals (
  goal_id text PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  goal_type text NOT NULL DEFAULT 'general',
  title text NOT NULL,
  target_calories numeric,
  target_protein_g numeric,
  target_carbs_g numeric,
  target_fat_g numeric,
  target_fiber_g numeric,
  target_water_ml numeric,
  target_weight_kg numeric,
  weekly_weight_delta_kg numeric,
  start_date date NOT NULL DEFAULT CURRENT_DATE,
  target_date date,
  status text NOT NULL DEFAULT 'active',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fitness.food_sources (
  source_key text PRIMARY KEY,
  name text NOT NULL,
  license text,
  attribution text,
  base_url text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO fitness.food_sources(source_key, name, license, attribution, base_url, metadata)
VALUES
  ('custom', 'Custom/manual foods', 'User-provided', 'User-entered data', NULL, '{}'::jsonb),
  ('open_food_facts', 'Open Food Facts', 'ODbL database / DbCL contents / CC BY-SA images', 'Data from Open Food Facts; comply with ODbL/DbCL and image attribution if media is used.', 'https://world.openfoodfacts.org', '{"rate_limits":{"product":"100/min","search":"10/min"}}'::jsonb),
  ('usda_fdc', 'USDA FoodData Central', 'US public-domain/federal data where applicable; verify branded-food terms', 'USDA FoodData Central', 'https://fdc.nal.usda.gov', '{}'::jsonb)
ON CONFLICT (source_key) DO UPDATE SET name=EXCLUDED.name, license=EXCLUDED.license, attribution=EXCLUDED.attribution, base_url=EXCLUDED.base_url, metadata=EXCLUDED.metadata;

CREATE TABLE IF NOT EXISTS fitness.foods (
  food_id text PRIMARY KEY,
  source_key text NOT NULL DEFAULT 'custom' REFERENCES fitness.food_sources(source_key),
  external_id text,
  barcode text,
  name text NOT NULL,
  brand text,
  category text,
  serving_size_g numeric,
  calories_per_100g numeric,
  protein_g_per_100g numeric,
  carbs_g_per_100g numeric,
  fat_g_per_100g numeric,
  fiber_g_per_100g numeric,
  sugar_g_per_100g numeric,
  sodium_mg_per_100g numeric,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source_key, external_id),
  UNIQUE (barcode)
);

CREATE TABLE IF NOT EXISTS fitness.nutrition_logs (
  nutrition_log_id text PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  food_id text REFERENCES fitness.foods(food_id) ON DELETE SET NULL,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  meal_type text NOT NULL DEFAULT 'unspecified',
  description text NOT NULL,
  quantity numeric NOT NULL DEFAULT 1,
  unit text NOT NULL DEFAULT 'serving',
  grams numeric,
  calories numeric,
  protein_g numeric,
  carbs_g numeric,
  fat_g numeric,
  fiber_g numeric,
  sugar_g numeric,
  sodium_mg numeric,
  water_ml numeric,
  source_confidence numeric NOT NULL DEFAULT 1,
  notes text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fitness.exercise_sources (
  source_key text PRIMARY KEY,
  name text NOT NULL,
  license text,
  attribution text,
  base_url text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO fitness.exercise_sources(source_key, name, license, attribution, base_url, metadata)
VALUES
  ('custom', 'Custom/manual exercises', 'User-provided', 'User-entered data', NULL, '{}'::jsonb),
  ('free_exercise_db', 'free-exercise-db compatible seed', 'Unlicense/public-domain style; verify media provenance separately', 'Exercise taxonomy inspired by free-exercise-db.', 'https://github.com/yuhonas/free-exercise-db', '{}'::jsonb)
ON CONFLICT (source_key) DO UPDATE SET name=EXCLUDED.name, license=EXCLUDED.license, attribution=EXCLUDED.attribution, base_url=EXCLUDED.base_url, metadata=EXCLUDED.metadata;

CREATE TABLE IF NOT EXISTS fitness.exercises (
  exercise_id text PRIMARY KEY,
  source_key text NOT NULL DEFAULT 'custom' REFERENCES fitness.exercise_sources(source_key),
  external_id text,
  name text NOT NULL,
  aliases jsonb NOT NULL DEFAULT '[]'::jsonb,
  category text,
  movement_pattern text,
  primary_muscles jsonb NOT NULL DEFAULT '[]'::jsonb,
  secondary_muscles jsonb NOT NULL DEFAULT '[]'::jsonb,
  equipment jsonb NOT NULL DEFAULT '[]'::jsonb,
  difficulty text,
  instructions jsonb NOT NULL DEFAULT '[]'::jsonb,
  media_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  contraindications jsonb NOT NULL DEFAULT '[]'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source_key, external_id)
);

INSERT INTO fitness.exercises(exercise_id, source_key, external_id, name, category, movement_pattern, primary_muscles, secondary_muscles, equipment, difficulty, instructions, metadata)
VALUES
  ('exercise-bodyweight-squat', 'custom', 'seed-bodyweight-squat', 'Bodyweight Squat', 'strength', 'squat', '["quadriceps","glutes"]'::jsonb, '["hamstrings","core"]'::jsonb, '["bodyweight"]'::jsonb, 'beginner', '["Stand with feet around shoulder width.","Sit hips down and back while keeping chest tall.","Drive through mid-foot to stand up."]'::jsonb, '{"seed":"fitness_core_v1"}'::jsonb),
  ('exercise-push-up', 'custom', 'seed-push-up', 'Push-up', 'strength', 'horizontal_push', '["chest","triceps"]'::jsonb, '["shoulders","core"]'::jsonb, '["bodyweight"]'::jsonb, 'beginner', '["Start in a plank with hands under shoulders.","Lower under control until chest approaches floor.","Push away while keeping body rigid."]'::jsonb, '{"seed":"fitness_core_v1"}'::jsonb),
  ('exercise-plank', 'custom', 'seed-plank', 'Plank', 'core', 'anti_extension', '["core"]'::jsonb, '["glutes","shoulders"]'::jsonb, '["bodyweight"]'::jsonb, 'beginner', '["Brace abdomen and glutes.","Keep ribs and pelvis stacked.","Hold without letting low back sag."]'::jsonb, '{"seed":"fitness_core_v1"}'::jsonb),
  ('exercise-dumbbell-row', 'custom', 'seed-dumbbell-row', 'Dumbbell Row', 'strength', 'horizontal_pull', '["back","lats"]'::jsonb, '["biceps","rear_delts"]'::jsonb, '["dumbbell"]'::jsonb, 'beginner', '["Hinge and support torso.","Pull elbow toward hip.","Lower under control."]'::jsonb, '{"seed":"fitness_core_v1"}'::jsonb),
  ('exercise-brisk-walk', 'custom', 'seed-brisk-walk', 'Brisk Walk', 'cardio', 'gait', '["cardiovascular"]'::jsonb, '["calves","glutes"]'::jsonb, '["none"]'::jsonb, 'beginner', '["Walk at a pace where talking is possible but singing is difficult.","Keep posture tall and cadence steady."]'::jsonb, '{"seed":"fitness_core_v1"}'::jsonb)
ON CONFLICT (exercise_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS fitness.routines (
  routine_id text PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  title text NOT NULL,
  goal_type text NOT NULL DEFAULT 'general',
  split_type text,
  days_per_week integer,
  status text NOT NULL DEFAULT 'active',
  plan jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fitness.routine_days (
  routine_day_id bigserial PRIMARY KEY,
  routine_id text NOT NULL REFERENCES fitness.routines(routine_id) ON DELETE CASCADE,
  day_index integer NOT NULL,
  title text NOT NULL,
  notes text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (routine_id, day_index)
);

CREATE TABLE IF NOT EXISTS fitness.routine_exercises (
  routine_exercise_id bigserial PRIMARY KEY,
  routine_day_id bigint NOT NULL REFERENCES fitness.routine_days(routine_day_id) ON DELETE CASCADE,
  exercise_id text REFERENCES fitness.exercises(exercise_id) ON DELETE SET NULL,
  order_index integer NOT NULL,
  sets integer,
  target_reps_min integer,
  target_reps_max integer,
  target_rpe numeric,
  target_rir numeric,
  rest_seconds integer,
  tempo text,
  progression_rule jsonb NOT NULL DEFAULT '{}'::jsonb,
  notes text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS fitness.workout_sessions (
  session_id text PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  routine_id text REFERENCES fitness.routines(routine_id) ON DELETE SET NULL,
  routine_day_id bigint REFERENCES fitness.routine_days(routine_day_id) ON DELETE SET NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  ended_at timestamptz,
  activity_type text NOT NULL DEFAULT 'strength',
  title text,
  status text NOT NULL DEFAULT 'in_progress',
  duration_minutes numeric,
  distance_km numeric,
  calories_burned numeric,
  perceived_effort numeric,
  readiness_score numeric,
  notes text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fitness.workout_sets (
  workout_set_id bigserial PRIMARY KEY,
  session_id text NOT NULL REFERENCES fitness.workout_sessions(session_id) ON DELETE CASCADE,
  exercise_id text NOT NULL REFERENCES fitness.exercises(exercise_id) ON DELETE CASCADE,
  set_index integer NOT NULL DEFAULT 1,
  set_type text NOT NULL DEFAULT 'working',
  weight_kg numeric,
  reps integer,
  duration_seconds integer,
  distance_m numeric,
  rpe numeric,
  rir numeric,
  completed boolean NOT NULL DEFAULT true,
  failure boolean NOT NULL DEFAULT false,
  rest_seconds integer,
  volume_load numeric NOT NULL DEFAULT 0,
  estimated_1rm numeric,
  notes text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fitness.body_metrics (
  body_metric_id bigserial PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  measured_at timestamptz NOT NULL DEFAULT now(),
  weight_kg numeric,
  body_fat_pct numeric,
  waist_cm numeric,
  chest_cm numeric,
  hip_cm numeric,
  arm_cm numeric,
  thigh_cm numeric,
  neck_cm numeric,
  resting_hr_bpm numeric,
  sleep_hours numeric,
  mood text,
  energy_level numeric,
  stress_level numeric,
  source text NOT NULL DEFAULT 'manual',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS fitness.coach_checkins (
  checkin_id text PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  summary text NOT NULL,
  sleep_quality numeric,
  soreness numeric,
  stress_level numeric,
  hunger_level numeric,
  energy_level numeric,
  adherence_score numeric,
  blockers text,
  next_steps text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fitness.coach_recommendations (
  coach_recommendation_id bigserial PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  generated_at timestamptz NOT NULL DEFAULT now(),
  category text NOT NULL,
  recommendation text NOT NULL,
  rationale text,
  status text NOT NULL DEFAULT 'generated',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS fitness.activity_imports (
  activity_import_id bigserial PRIMARY KEY,
  profile_id text NOT NULL REFERENCES fitness.profiles(profile_id) ON DELETE CASCADE,
  source text NOT NULL,
  external_id text,
  file_ref text,
  activity_type text,
  started_at timestamptz,
  duration_seconds integer,
  distance_m numeric,
  elevation_gain_m numeric,
  calories_estimated numeric,
  avg_hr_bpm numeric,
  max_hr_bpm numeric,
  streams jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fitness_profiles_owner ON fitness.profiles(owner_id);
CREATE INDEX IF NOT EXISTS idx_fitness_goals_profile_status ON fitness.goals(profile_id, status, start_date DESC);
CREATE INDEX IF NOT EXISTS idx_fitness_foods_name ON fitness.foods USING gin (to_tsvector('simple', name));
CREATE INDEX IF NOT EXISTS idx_fitness_foods_barcode ON fitness.foods(barcode);
CREATE INDEX IF NOT EXISTS idx_fitness_nutrition_profile_time ON fitness.nutrition_logs(profile_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_fitness_exercises_name ON fitness.exercises USING gin (to_tsvector('simple', name));
CREATE INDEX IF NOT EXISTS idx_fitness_routines_profile_status ON fitness.routines(profile_id, status);
CREATE INDEX IF NOT EXISTS idx_fitness_workout_sessions_profile_time ON fitness.workout_sessions(profile_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_fitness_workout_sets_session ON fitness.workout_sets(session_id, exercise_id, set_index);
CREATE INDEX IF NOT EXISTS idx_fitness_body_metrics_profile_time ON fitness.body_metrics(profile_id, measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_fitness_checkins_profile_time ON fitness.coach_checkins(profile_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_fitness_recommendations_profile_time ON fitness.coach_recommendations(profile_id, generated_at DESC);

GRANT USAGE ON SCHEMA fitness TO fitness_runtime, agent_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA fitness TO fitness_runtime;
GRANT SELECT ON ALL TABLES IN SCHEMA fitness TO agent_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA fitness TO fitness_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA fitness GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fitness_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA fitness GRANT SELECT ON TABLES TO agent_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA fitness GRANT USAGE, SELECT ON SEQUENCES TO fitness_runtime;
