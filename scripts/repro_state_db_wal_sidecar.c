#define _GNU_SOURCE
#include <sqlite3.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_RESULT 512
#define BUSY_TIMEOUT_MS 250

typedef struct {
  long ok;
  long busy;
  long errors;
} child_stats;

typedef struct {
  dev_t dev;
  ino_t ino;
  int present;
} inode_snapshot;

static void sleep_ms(int ms) {
  struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (long)(ms % 1000) * 1000000L};
  while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

static int exec_sql(sqlite3 *db, const char *sql, char **errmsg_out) {
  char *errmsg = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &errmsg);
  if (rc != SQLITE_OK && errmsg_out != NULL) {
    *errmsg_out = errmsg;
  } else if (errmsg != NULL) {
    sqlite3_free(errmsg);
  }
  return rc;
}

static int open_db(const char *path, sqlite3 **db_out) {
  sqlite3 *db = NULL;
  int rc = sqlite3_open_v2(
      path,
      &db,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
      NULL);
  if (rc != SQLITE_OK) {
    if (db != NULL) sqlite3_close(db);
    return rc;
  }
  sqlite3_busy_timeout(db, BUSY_TIMEOUT_MS);
  *db_out = db;
  return SQLITE_OK;
}

static int init_db(const char *path) {
  sqlite3 *db = NULL;
  int rc = open_db(path, &db);
  if (rc != SQLITE_OK) return rc;
  const char *sql =
      "PRAGMA journal_mode=WAL;"
      "PRAGMA synchronous=FULL;"
      "PRAGMA wal_autocheckpoint=0;"
      "CREATE TABLE IF NOT EXISTS messages("
      " id INTEGER PRIMARY KEY AUTOINCREMENT,"
      " content TEXT NOT NULL"
      ");"
      "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5("
      " content, content='messages', content_rowid='id'"
      ");"
      "CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN "
      " INSERT INTO messages_fts(rowid, content) VALUES(new.id, new.content); END;"
      "CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN "
      " INSERT INTO messages_fts(messages_fts, rowid, content) "
      " VALUES('delete', old.id, old.content); END;"
      "CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN "
      " INSERT INTO messages_fts(messages_fts, rowid, content) "
      " VALUES('delete', old.id, old.content);"
      " INSERT INTO messages_fts(rowid, content) VALUES(new.id, new.content); END;"
      "INSERT INTO messages(content) VALUES('seed');"
      "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');";
  char *errmsg = NULL;
  rc = exec_sql(db, sql, &errmsg);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "init failed: %s\n", errmsg ? errmsg : sqlite3_errmsg(db));
    sqlite3_free(errmsg);
  }
  sqlite3_close(db);
  return rc;
}

static int is_busy_rc(int rc) {
  int primary = rc & 0xff;
  return primary == SQLITE_BUSY || primary == SQLITE_LOCKED;
}

static void writer_process(const char *path, int ready_fd, int stop_fd, int stats_fd) {
  child_stats stats = {0};
  sqlite3 *db = NULL;
  if (open_db(path, &db) != SQLITE_OK) {
    stats.errors++;
    write(stats_fd, &stats, sizeof(stats));
    _exit(2);
  }
  char *errmsg = NULL;
  int rc = exec_sql(
      db,
      "PRAGMA journal_mode=WAL;"
      "PRAGMA synchronous=FULL;"
      "PRAGMA wal_autocheckpoint=0;",
      &errmsg);
  if (rc != SQLITE_OK) {
    stats.errors++;
    sqlite3_free(errmsg);
  }
  int flags = fcntl(stop_fd, F_GETFL, 0);
  fcntl(stop_fd, F_SETFL, flags | O_NONBLOCK);
  write(ready_fd, "R", 1);

  sqlite3_stmt *stmt = NULL;
  rc = sqlite3_prepare_v2(
      db, "INSERT INTO messages(content) VALUES(?)", -1, &stmt, NULL);
  if (rc != SQLITE_OK) stats.errors++;

  long sequence = 0;
  for (;;) {
    char stop;
    if (read(stop_fd, &stop, 1) == 1) break;

    rc = exec_sql(db, "BEGIN IMMEDIATE", NULL);
    if (rc != SQLITE_OK) {
      if (is_busy_rc(rc)) stats.busy++;
      else stats.errors++;
      sleep_ms(1);
      continue;
    }

    char payload[96];
    snprintf(
        payload,
        sizeof(payload),
        "writer=%ld pid=%ld",
        sequence++,
        (long)getpid());
    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
    sqlite3_bind_text(stmt, 1, payload, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_DONE) {
      int commit_rc = exec_sql(db, "COMMIT", NULL);
      if (commit_rc == SQLITE_OK) stats.ok++;
      else {
        if (is_busy_rc(commit_rc)) stats.busy++;
        else stats.errors++;
        exec_sql(db, "ROLLBACK", NULL);
      }
    } else {
      if (is_busy_rc(rc)) stats.busy++;
      else stats.errors++;
      exec_sql(db, "ROLLBACK", NULL);
    }

    if ((sequence % 64) == 0) {
      int log_frames = 0;
      int checkpointed = 0;
      int ck = sqlite3_wal_checkpoint_v2(
          db,
          NULL,
          SQLITE_CHECKPOINT_PASSIVE,
          &log_frames,
          &checkpointed);
      if (ck != SQLITE_OK && !is_busy_rc(ck)) stats.errors++;
    }
  }

  sqlite3_finalize(stmt);
  sqlite3_close(db);
  write(stats_fd, &stats, sizeof(stats));
  _exit(stats.errors ? 1 : 0);
}

static int recreate_fts(sqlite3 *db) {
  const char *sql =
      "BEGIN IMMEDIATE;"
      "DROP TRIGGER IF EXISTS messages_ai;"
      "DROP TRIGGER IF EXISTS messages_ad;"
      "DROP TRIGGER IF EXISTS messages_au;"
      "DROP TABLE IF EXISTS messages_fts;"
      "CREATE VIRTUAL TABLE messages_fts USING fts5("
      " content, content='messages', content_rowid='id'"
      ");"
      "CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN "
      " INSERT INTO messages_fts(rowid, content) VALUES(new.id, new.content); END;"
      "CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN "
      " INSERT INTO messages_fts(messages_fts, rowid, content) "
      " VALUES('delete', old.id, old.content); END;"
      "CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN "
      " INSERT INTO messages_fts(messages_fts, rowid, content) "
      " VALUES('delete', old.id, old.content);"
      " INSERT INTO messages_fts(rowid, content) VALUES(new.id, new.content); END;"
      "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');"
      "COMMIT;";
  int rc = exec_sql(db, sql, NULL);
  if (rc != SQLITE_OK) exec_sql(db, "ROLLBACK", NULL);
  return rc;
}

static void maintenance_process(
    const char *path,
    const char *scenario,
    int loops,
    int start_fd,
    int stats_fd) {
  child_stats stats = {0};
  char start;
  if (read(start_fd, &start, 1) != 1) {
    stats.errors++;
    write(stats_fd, &stats, sizeof(stats));
    _exit(2);
  }

  char wal_path[4096];
  char shm_path[4096];
  snprintf(wal_path, sizeof(wal_path), "%s-wal", path);
  snprintf(shm_path, sizeof(shm_path), "%s-shm", path);

  for (int i = 0; i < loops; i++) {
    if (strcmp(scenario, "forced_unlink") == 0 && i == 0) {
      if (unlink(wal_path) != 0 && errno != ENOENT) stats.errors++;
      if (unlink(shm_path) != 0 && errno != ENOENT) stats.errors++;
      sleep_ms(5);
    }

    sqlite3 *db = NULL;
    int rc = open_db(path, &db);
    if (rc != SQLITE_OK) {
      if (is_busy_rc(rc)) stats.busy++;
      else stats.errors++;
      sleep_ms(2);
      continue;
    }

    if (strcmp(scenario, "open_probe") == 0) {
      rc = exec_sql(
          db,
          "PRAGMA journal_mode;"
          "PRAGMA wal_checkpoint(PASSIVE);"
          "SELECT count(*) FROM messages;",
          NULL);
    } else if (strcmp(scenario, "set_wal") == 0) {
      rc = exec_sql(
          db,
          "PRAGMA journal_mode=WAL;"
          "PRAGMA wal_checkpoint(PASSIVE);",
          NULL);
    } else if (strcmp(scenario, "fts_rebuild") == 0) {
      rc = exec_sql(
          db,
          "BEGIN IMMEDIATE;"
          "INSERT INTO messages_fts(messages_fts) VALUES('rebuild');"
          "COMMIT;",
          NULL);
      if (rc != SQLITE_OK) exec_sql(db, "ROLLBACK", NULL);
    } else if (strcmp(scenario, "drop_recreate_fts") == 0) {
      rc = recreate_fts(db);
    } else if (strcmp(scenario, "forced_unlink") == 0) {
      rc = exec_sql(
          db,
          "PRAGMA journal_mode=WAL;"
          "BEGIN IMMEDIATE;"
          "INSERT INTO messages(content) VALUES('forced-unlink-writer');"
          "COMMIT;"
          "PRAGMA wal_checkpoint(PASSIVE);",
          NULL);
      if (rc != SQLITE_OK) exec_sql(db, "ROLLBACK", NULL);
    } else {
      rc = SQLITE_MISUSE;
    }

    if (rc == SQLITE_OK) stats.ok++;
    else if (is_busy_rc(rc)) stats.busy++;
    else stats.errors++;
    sqlite3_close(db);
    sleep_ms(2);
  }

  write(stats_fd, &stats, sizeof(stats));
  _exit(stats.errors ? 1 : 0);
}

static inode_snapshot snapshot_inode(const char *path) {
  struct stat st;
  inode_snapshot snap = {0};
  if (stat(path, &st) == 0) {
    snap.dev = st.st_dev;
    snap.ino = st.st_ino;
    snap.present = 1;
  }
  return snap;
}

static int inode_replaced(inode_snapshot before, inode_snapshot after) {
  return before.present && after.present &&
      (before.dev != after.dev || before.ino != after.ino);
}

static int process_has_deleted_sidecar(pid_t pid, const char *db_path) {
  char fd_dir[128];
  snprintf(fd_dir, sizeof(fd_dir), "/proc/%ld/fd", (long)pid);
  DIR *dir = opendir(fd_dir);
  if (dir == NULL) return 0;

  char wal_needle[4096];
  char shm_needle[4096];
  snprintf(wal_needle, sizeof(wal_needle), "%s-wal", db_path);
  snprintf(shm_needle, sizeof(shm_needle), "%s-shm", db_path);
  int found = 0;
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_name[0] == '.') continue;
    char link_path[512];
    char target[8192];
    snprintf(link_path, sizeof(link_path), "%s/%s", fd_dir, entry->d_name);
    ssize_t n = readlink(link_path, target, sizeof(target) - 1);
    if (n < 0) continue;
    target[n] = '\0';
    if (strstr(target, " (deleted)") != NULL &&
        (strstr(target, wal_needle) != NULL ||
         strstr(target, shm_needle) != NULL)) {
      found = 1;
      break;
    }
  }
  closedir(dir);
  return found;
}

static int scalar_text(sqlite3 *db, const char *sql, char *out, size_t out_size) {
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    const unsigned char *value = sqlite3_column_text(stmt, 0);
    snprintf(out, out_size, "%s", value ? (const char *)value : "null");
    rc = SQLITE_OK;
  }
  sqlite3_finalize(stmt);
  return rc;
}

static long long scalar_int64(sqlite3 *db, const char *sql, long long fallback) {
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) {
    return fallback;
  }
  long long result = fallback;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    result = sqlite3_column_int64(stmt, 0);
  }
  sqlite3_finalize(stmt);
  return result;
}

static void json_escape(
    const char *input,
    char *output,
    size_t output_size) {
  size_t j = 0;
  if (output_size == 0) return;
  for (size_t i = 0; input && input[i] && j + 2 < output_size; i++) {
    unsigned char c = (unsigned char)input[i];
    if (c == '"' || c == '\\') {
      output[j++] = '\\';
      output[j++] = (char)c;
    } else if (c == '\n') {
      output[j++] = '\\';
      output[j++] = 'n';
    } else if (c >= 0x20) {
      output[j++] = (char)c;
    }
  }
  output[j] = '\0';
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s SCENARIO DB_PATH LOOPS\n", argv[0]);
    return 64;
  }
  const char *scenario = argv[1];
  const char *path = argv[2];
  int loops = atoi(argv[3]);
  if (loops <= 0) loops = 100;

  unlink(path);
  char wal_path[4096];
  char shm_path[4096];
  snprintf(wal_path, sizeof(wal_path), "%s-wal", path);
  snprintf(shm_path, sizeof(shm_path), "%s-shm", path);
  unlink(wal_path);
  unlink(shm_path);

  int init_rc = init_db(path);
  if (init_rc != SQLITE_OK) return 2;

  int ready_pipe[2];
  int stop_pipe[2];
  int writer_stats_pipe[2];
  int maint_start_pipe[2];
  int maint_stats_pipe[2];
  if (pipe(ready_pipe) || pipe(stop_pipe) || pipe(writer_stats_pipe) ||
      pipe(maint_start_pipe) || pipe(maint_stats_pipe)) {
    perror("pipe");
    return 2;
  }

  pid_t writer = fork();
  if (writer == 0) {
    close(ready_pipe[0]);
    close(stop_pipe[1]);
    close(writer_stats_pipe[0]);
    close(maint_start_pipe[0]);
    close(maint_start_pipe[1]);
    close(maint_stats_pipe[0]);
    close(maint_stats_pipe[1]);
    writer_process(path, ready_pipe[1], stop_pipe[0], writer_stats_pipe[1]);
  }
  close(ready_pipe[1]);
  close(stop_pipe[0]);
  close(writer_stats_pipe[1]);
  char ready;
  if (read(ready_pipe[0], &ready, 1) != 1) {
    kill(writer, SIGKILL);
    return 2;
  }
  close(ready_pipe[0]);

  pid_t maint = fork();
  if (maint == 0) {
    close(maint_start_pipe[1]);
    close(maint_stats_pipe[0]);
    close(stop_pipe[1]);
    close(writer_stats_pipe[0]);
    maintenance_process(
        path,
        scenario,
        loops,
        maint_start_pipe[0],
        maint_stats_pipe[1]);
  }
  close(maint_start_pipe[0]);
  close(maint_stats_pipe[1]);
  write(maint_start_pipe[1], "S", 1);
  close(maint_start_pipe[1]);

  inode_snapshot wal_prev = snapshot_inode(wal_path);
  inode_snapshot shm_prev = snapshot_inode(shm_path);
  long wal_replacements = 0;
  long shm_replacements = 0;
  int deleted_fd_observed = 0;
  int maint_status = 0;
  for (;;) {
    inode_snapshot wal_now = snapshot_inode(wal_path);
    inode_snapshot shm_now = snapshot_inode(shm_path);
    if (inode_replaced(wal_prev, wal_now)) wal_replacements++;
    if (inode_replaced(shm_prev, shm_now)) shm_replacements++;
    if (wal_now.present) wal_prev = wal_now;
    if (shm_now.present) shm_prev = shm_now;
    if (process_has_deleted_sidecar(writer, path) ||
        process_has_deleted_sidecar(maint, path)) {
      deleted_fd_observed = 1;
    }
    pid_t done = waitpid(maint, &maint_status, WNOHANG);
    if (done == maint) break;
    sleep_ms(1);
  }

  write(stop_pipe[1], "X", 1);
  close(stop_pipe[1]);
  int writer_status = 0;
  waitpid(writer, &writer_status, 0);

  child_stats writer_stats = {0};
  child_stats maint_stats = {0};
  read(writer_stats_pipe[0], &writer_stats, sizeof(writer_stats));
  read(maint_stats_pipe[0], &maint_stats, sizeof(maint_stats));
  close(writer_stats_pipe[0]);
  close(maint_stats_pipe[0]);

  sqlite3 *db = NULL;
  char integrity[MAX_RESULT] = "unavailable";
  char quick[MAX_RESULT] = "unavailable";
  long long messages = -1;
  long long fts_rows = -1;
  int checkpoint_rc = -1;
  if (open_db(path, &db) == SQLITE_OK) {
    int log_frames = 0;
    int checkpointed = 0;
    checkpoint_rc = sqlite3_wal_checkpoint_v2(
        db,
        NULL,
        SQLITE_CHECKPOINT_TRUNCATE,
        &log_frames,
        &checkpointed);
    scalar_text(db, "PRAGMA integrity_check", integrity, sizeof(integrity));
    scalar_text(db, "PRAGMA quick_check", quick, sizeof(quick));
    messages = scalar_int64(db, "SELECT count(*) FROM messages", -1);
    fts_rows = scalar_int64(db, "SELECT count(*) FROM messages_fts", -1);
    sqlite3_close(db);
  }

  char integrity_json[MAX_RESULT * 2];
  char quick_json[MAX_RESULT * 2];
  json_escape(integrity, integrity_json, sizeof(integrity_json));
  json_escape(quick, quick_json, sizeof(quick_json));
  printf(
      "{\"sqlite_version\":\"%s\",\"sqlite_source_id\":\"%s\","
      "\"scenario\":\"%s\",\"loops\":%d,"
      "\"writer_ok\":%ld,\"writer_busy\":%ld,\"writer_errors\":%ld,"
      "\"maintenance_ok\":%ld,\"maintenance_busy\":%ld,"
      "\"maintenance_errors\":%ld,"
      "\"wal_inode_replacements\":%ld,"
      "\"shm_inode_replacements\":%ld,"
      "\"deleted_fd_observed\":%s,\"checkpoint_rc\":%d,"
      "\"integrity_check\":\"%s\",\"quick_check\":\"%s\","
      "\"message_count\":%lld,\"fts_row_count\":%lld,"
      "\"writer_exit\":%d,\"maintenance_exit\":%d}\n",
      sqlite3_libversion(),
      sqlite3_sourceid(),
      scenario,
      loops,
      writer_stats.ok,
      writer_stats.busy,
      writer_stats.errors,
      maint_stats.ok,
      maint_stats.busy,
      maint_stats.errors,
      wal_replacements,
      shm_replacements,
      deleted_fd_observed ? "true" : "false",
      checkpoint_rc,
      integrity_json,
      quick_json,
      messages,
      fts_rows,
      WIFEXITED(writer_status) ? WEXITSTATUS(writer_status) : 128,
      WIFEXITED(maint_status) ? WEXITSTATUS(maint_status) : 128);
  return 0;
}
