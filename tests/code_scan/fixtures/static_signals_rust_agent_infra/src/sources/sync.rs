// Remote source sync using ssh/rsync/sftp
pub fn sync_remote_ssh(host: &str) {
    // ssh2 used here
    println!("sync over ssh to {}", host);
}

pub fn sync_rsync(source: &str) {
    println!("rsync {}", source);
}

pub fn sync_sftp(remote: &str) {
    println!("sftp sync {}", remote);
}
