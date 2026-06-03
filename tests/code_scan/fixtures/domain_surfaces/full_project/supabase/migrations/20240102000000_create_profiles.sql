-- 20240102000000_create_profiles.sql
create table profiles (
  id uuid primary key references users(id),
  display_name text
);
