/* Copyright (c) 2008-2009, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ---
 * Author: Kostya Serebryany
 * Copied to CPython by Jeffrey Yasskin, with all macros renamed to
 * start with _Py_ to avoid colliding with users embedding Python, and
 * with deprecated macros removed.
 */

/* This file defines dynamic annotations for use with dynamic analysis
   tool such as valgrind, PIN, etc.

   Dynamic annotation is a source code annotation that affects
   the generated code (that is, the annotation is not a comment).
   Each such annotation is attached to a particular
   instruction and/or to a particular object (address) in the program.

   The annotations that should be used by users are macros in all upper-case
   (e.g., _Py_ANNOTATE_NEW_MEMORY).

   Actual implementation of these macros may differ depending on the
   dynamic analysis tool being used.

   See https://code.google.com/p/data-race-test/  for more information.

   This file supports the following dynamic analysis tools:
   - None (DYNAMIC_ANNOTATIONS_ENABLED is not defined or zero).
      Macros are defined empty.
   - ThreadSanitizer, Helgrind, DRD (DYNAMIC_ANNOTATIONS_ENABLED is 1).
      Macros are defined as calls to non-inlinable empty functions
      that are intercepted by Valgrind. */

#ifndef __DYNAMIC_ANNOTATIONS_H__
#define __DYNAMIC_ANNOTATIONS_H__

#ifndef DYNAMIC_ANNOTATIONS_ENABLED
# define DYNAMIC_ANNOTATIONS_ENABLED 0
#endif

#if DYNAMIC_ANNOTATIONS_ENABLED != 0

  /* -------------------------------------------------------------
     Annotations useful when implementing condition variables such as CondVar,
     using conditional critical sections (Await/LockWhen) and when constructing
     user-defined synchronization mechanisms.

     The annotations _Py_ANNOTATE_HAPPENS_BEFORE() and
     _Py_ANNOTATE_HAPPENS_AFTER() can be used to define happens-before arcs in
     user-defined synchronization mechanisms: the race detector will infer an
     arc from the former to the latter when they share the same argument
     pointer.

     Example 1 (reference counting):

     void Unref() {
       _Py_ANNOTATE_HAPPENS_BEFORE(&refcount_);
       if (AtomicDecrementByOne(&refcount_) == 0) {
         _Py_ANNOTATE_HAPPENS_AFTER(&refcount_);
         delete this;
       }
     }

     Example 2 (message queue):

     void MyQueue::Put(Type *e) {
       MutexLock lock(&mu_);
       _Py_ANNOTATE_HAPPENS_BEFORE(e);
       PutElementIntoMyQueue(e);
     }

     Type *MyQueue::Get() {
       MutexLock lock(&mu_);
       Type *e = GetElementFromMyQueue();
       _Py_ANNOTATE_HAPPENS_AFTER(e);
       return e;
     }

     Note: when possible, please use the existing reference counting and message
     queue implementations instead of inventing new ones. */

  /* Report that wait on the condition variable at address "cv" has succeeded
     and the lock at address "lock" is held. */
#define _Py_ANNOTATE_CONDVAR_LOCK_WAIT(cv, lock) \
    AnnotateCondVarWait(__FILE__, __LINE__, cv, lock)

  /* Report that wait on the condition variable at "cv" has succeeded.  Variant
     w/o lock. */
#define _Py_ANNOTATE_CONDVAR_WAIT(cv) \
    AnnotateCondVarWait(__FILE__, __LINE__, cv, NULL)

  /* Report that we are about to signal on the condition variable at address
     "cv". */
#define _Py_ANNOTATE_CONDVAR_SIGNAL(cv) \
    AnnotateCondVarSignal(__FILE__, __LINE__, cv)

  /* Report that we are about to signal_all on the condition variable at "cv". */
#define _Py_ANNOTATE_CONDVAR_SIGNAL_ALL(cv) \
    AnnotateCondVarSignalAll(__FILE__, __LINE__, cv)

  /* Annotations for user-defined synchronization mechanisms. */
#define _Py_ANNOTATE_HAPPENS_BEFORE(obj) _Py_ANNOTATE_CONDVAR_SIGNAL(obj)
#define _Py_ANNOTATE_HAPPENS_AFTER(obj)  _Py_ANNOTATE_CONDVAR_WAIT(obj)

  /* Report that the bytes in the range [pointer, pointer+size) are about
     to be published safely. The race checker will create a happens-before
     arc from the call _Py_ANNOTATE_PUBLISH_MEMORY_RANGE(pointer, size) to
     subsequent accesses to this memory.
     Note: this annotation may not work properly if the race detector uses
     sampling, i.e. does not observe all memory accesses.
     */
#define _Py_ANNOTATE_PUBLISH_MEMORY_RANGE(pointer, size) \
    AnnotatePublishMemoryRange(__FILE__, __LINE__, pointer, size)

  /* Instruct the tool to create a happens-before arc between mu->Unlock() and
     mu->Lock(). This annotation may slow down the race detector and hide real
     races. Normally it is used only when it would be difficult to annotate each
     of the mutex's critical sections individually using the annotations above.
     This annotation makes sense only for hybrid race detectors. For pure
     happens-before detectors this is a no-op. For more details see
     https://code.google.com/p/data-race-test/wiki/PureHappensBeforeVsHybrid . */
#define _Py_ANNOTATE_PURE_HAPPENS_BEFORE_MUTEX(mu) \
    AnnotateMutexIsUsedAsCondVar(__FILE__, __LINE__, mu)

  /* -------------------------------------------------------------
     Annotations useful when defining memory allocators, or when memory that
     was protected in one way starts to be protected in another. */

  /* Report that a new memory at "address" of size "size" has been allocated.
     This might be used when the memory has been retrieved from a free list and
     is about to be reused, or when the locking discipline for a variable
     changes. */
#define _Py_ANNOTATE_NEW_MEMORY(address, size) \
    AnnotateNewMemory(__FILE__, __LINE__, address, size)

  /* -------------------------------------------------------------
     Annotations useful when defining FIFO queues that transfer data between
     threads. */

  /* Report that the producer-consumer queue (such as ProducerConsumerQueue) at
     address "pcq" has been created.  The _Py_ANNOTATE_PCQ_* annotations should
     be used only for FIFO queues.  For non-FIFO queues use
     _Py_ANNOTATE_HAPPENS_BEFORE (for put) and _Py_ANNOTATE_HAPPENS_AFTER (for
     get). */
#define _Py_ANNOTATE_PCQ_CREATE(pcq) \
    AnnotatePCQCreate(__FILE__, __LINE__, pcq)

  /* Report that the queue at address "pcq" is about to be destroyed. */
#define _Py_ANNOTATE_PCQ_DESTROY(pcq) \
    AnnotatePCQDestroy(__FILE__, __LINE__, pcq)

  /* Report that we are about to put an element into a FIFO queue at address
     "pcq". */
#define _Py_ANNOTATE_PCQ_PUT(pcq) \
    AnnotatePCQPut(__FILE__, __LINE__, pcq)

  /* Report that we've just got an element from a FIFO queue at address "pcq". */
#define _Py_ANNOTATE_PCQ_GET(pcq) \
    AnnotatePCQGet(__FILE__, __LINE__, pcq)

  /* -------------------------------------------------------------
     Annotations that suppress errors.  It is usually better to express the
     program's synchronization using the other annotations, but these can
     be used when all else fails. */

  /* Report that we may have a benign race at "pointer", with size
     "sizeof(*(pointer))". "pointer" must be a non-void* pointer.  Insert at the
     point where "pointer" has been allocated, preferably close to the point
     where the race happens.  See also _Py_ANNOTATE_BENIGN_RACE_STATIC. */
#define _Py_ANNOTATE_BENIGN_RACE(pointer, description) \
    AnnotateBenignRaceSized(__FILE__, __LINE__, pointer, \
                            sizeof(*(pointer)), description)

  /* Same as _Py_ANNOTATE_BENIGN_RACE(address, description), but applies to
     the memory range [address, address+size). */
#define _Py_ANNOTATE_BENIGN_RACE_SIZED(address, size, description) \
    AnnotateBenignRaceSized(__FILE__, __LINE__, address, size, description)

  /* Request the analysis tool to ignore all reads in the current thread
     until _Py_ANNOTATE_IGNORE_READS_END is called.
     Useful to ignore intentional racey reads, while still checking
     other reads and all writes.
     See also _Py_ANNOTATE_UNPROTECTED_READ. */
#define _Py_ANNOTATE_IGNORE_READS_BEGIN() \
    AnnotateIgnoreReadsBegin(__FILE__, __LINE__)

  /* Stop ignoring reads. */
#define _Py_ANNOTATE_IGNORE_READS_END() \
    AnnotateIgnoreReadsEnd(__FILE__, __LINE__)

  /* Similar to _Py_ANNOTATE_IGNORE_READS_BEGIN, but ignore writes. */
#define _Py_ANNOTATE_IGNORE_WRITES_BEGIN() \
    AnnotateIgnoreWritesBegin(__FILE__, __LINE__)

  /* Stop ignoring writes. */
#define _Py_ANNOTATE_IGNORE_WRITES_END() \
    AnnotateIgnoreWritesEnd(__FILE__, __LINE__)

  /* Start ignoring all memory accesses (reads and writes). */
#define _Py_ANNOTATE_IGNORE_READS_AND_WRITES_BEGIN() \
    do {\
      _Py_ANNOTATE_IGNORE_READS_BEGIN();\
      _Py_ANNOTATE_IGNORE_WRITES_BEGIN();\
    }while(0)\

  /* Stop ignoring all memory accesses. */
#define _Py_ANNOTATE_IGNORE_READS_AND_WRITES_END() \
    do {\
      _Py_ANNOTATE_IGNORE_WRITES_END();\
      _Py_ANNOTATE_IGNORE_READS_END();\
    }while(0)\

  /* Similar to _Py_ANNOTATE_IGNORE_READS_BEGIN, but ignore synchronization events:
     RWLOCK* and CONDVAR*. */
#define _Py_ANNOTATE_IGNORE_SYNC_BEGIN() \
    AnnotateIgnoreSyncBegin(__FILE__, __LINE__)

  /* Stop ignoring sync events. */
#define _Py_ANNOTATE_IGNORE_SYNC_END() \
    AnnotateIgnoreSyncEnd(__FILE__, __LINE__)


  /* Enable (enable!=0) or disable (enable==0) race detection for all threads.
     This annotation could be useful if you want to skip expensive race analysis
     during some period of program execution, e.g. during initialization. */
#define _Py_ANNOTATE_ENABLE_RACE_DETECTION(enable) \
    AnnotateEnableRaceDetection(__FILE__, __LINE__, enable)

  /* -------------------------------------------------------------
     Annotations useful for debugging. */

  /* Request to trace every access to "address". */
#define _Py_ANNOTATE_TRACE_MEMORY(address) \
    AnnotateTraceMemory(__FILE__, __LINE__, address)

  /* Report the current thread name to a race detector. */
#define _Py_ANNOTATE_THREAD_NAME(name) \
    AnnotateThreadName(__FILE__, __LINE__, name)

  /* -------------------------------------------------------------
     Annotations useful when implementing locks.  They are not
     normally needed by modules that merely use locks.
     The "lock" argument is a pointer to the lock object. */

  /* Report that a lock has been created at address "lock". */
#define _Py_ANNOTATE_RWLOCK_CREATE(lock) \
    AnnotateRWLockCreate(__FILE__, __LINE__, lock)

  /* Report that the lock at address "lock" is about to be destroyed. */
#define _Py_ANNOTATE_RWLOCK_DESTROY(lock) \
    AnnotateRWLockDestroy(__FILE__, __LINE__, lock)

  /* Report that the lock at address "lock" has been acquired.
     is_w=1 for writer lock, is_w=0 for reader lock. */
#define _Py_ANNOTATE_RWLOCK_ACQUIRED(lock, is_w) \
    AnnotateRWLockAcquired(__FILE__, __LINE__, lock, is_w)

  /* Report that the lock at address "lock" is about to be released. */
#define _Py_ANNOTATE_RWLOCK_RELEASED(lock, is_w) \
    AnnotateRWLockReleased(__FILE__, __LINE__, lock, is_w)

  /* -------------------------------------------------------------
     Annotations useful when implementing barriers.  They are not
     normally needed by modules that merely use barriers.
     The "barrier" argument is a pointer to the barrier object. */

  /* Report that the "barrier" has been initialized with initial "count".
   If 'reinitialization_allowed' is true, initialization is allowed to happen
   multiple times w/o calling barrier_destroy() */
#define _Py_ANNOTATE_BARRIER_INIT(barrier, count, reinitialization_allowed) \
    AnnotateBarrierInit(__FILE__, __LINE__, barrier, count, \
                        reinitialization_allowed)

  /* Report that we are about to enter barrier_wait("barrier"). */
#define _Py_ANNOTATE_BARRIER_WAIT_BEFORE(barrier) \
    AnnotateBarrierWaitBefore(__FILE__, __LINE__, barrier)

  /* Report that we just exited barrier_wait("barrier"). */
#define _Py_ANNOTATE_BARRIER_WAIT_AFTER(barrier) \
    AnnotateBarrierWaitAfter(__FILE__, __LINE__, barrier)

  /* Report that the "barrier" has been destroyed. */
#define _Py_ANNOTATE_BARRIER_DESTROY(barrier) \
    AnnotateBarrierDestroy(__FILE__, __LINE__, barrier)

  /* -------------------------------------------------------------
     Annotations useful for testing race detectors. */

  /* Report that we expect a race on the variable at "address".
     Use only in unit tests for a race detector. */
#define _Py_ANNOTATE_EXPECT_RACE(address, description) \
    AnnotateExpectRace(__FILE__, __LINE__, address, description)

  /* A no-op. Insert where you like to test the interceptors. */
#define _Py_ANNOTATE_NO_OP(arg) \
    AnnotateNoOp(__FILE__, __LINE__, arg)

  /* Force the race detector to flush its state. The actual effect depends on
   * the implementation of the detector. */
#define _Py_ANNOTATE_FLUSH_STATE() \
    AnnotateFlushState(__FILE__, __LINE__)


#else  /* DYNAMIC_ANNOTATIONS_ENABLED == 0 */

#define _Py_ANNOTATE_RWLOCK_CREATE(lock) /* empty */
#define _Py_ANNOTATE_RWLOCK_DESTROY(lock) /* empty */
#define _Py_ANNOTATE_RWLOCK_ACQUIRED(lock, is_w) /* empty */
#define _Py_ANNOTATE_RWLOCK_RELEASED(lock, is_w) /* empty */
#define _Py_ANNOTATE_BARRIER_INIT(barrier, count, reinitialization_allowed) /* */
#define _Py_ANNOTATE_BARRIER_WAIT_BEFORE(barrier) /* empty */
#define _Py_ANNOTATE_BARRIER_WAIT_AFTER(barrier) /* empty */
#define _Py_ANNOTATE_BARRIER_DESTROY(barrier) /* empty */
#define _Py_ANNOTATE_CONDVAR_LOCK_WAIT(cv, lock) /* empty */
#define _Py_ANNOTATE_CONDVAR_WAIT(cv) /* empty */
#define _Py_ANNOTATE_CONDVAR_SIGNAL(cv) /* empty */
#define _Py_ANNOTATE_CONDVAR_SIGNAL_ALL(cv) /* empty */
#define _Py_ANNOTATE_HAPPENS_BEFORE(obj) /* empty */
#define _Py_ANNOTATE_HAPPENS_AFTER(obj) /* empty */
#define _Py_ANNOTATE_PUBLISH_MEMORY_RANGE(address, size) /* empty */
#define _Py_ANNOTATE_UNPUBLISH_MEMORY_RANGE(address, size)  /* empty */
#define _Py_ANNOTATE_SWAP_MEMORY_RANGE(address, size)  /* empty */
#define _Py_ANNOTATE_PCQ_CREATE(pcq) /* empty */
#define _Py_ANNOTATE_PCQ_DESTROY(pcq) /* empty */
#define _Py_ANNOTATE_PCQ_PUT(pcq) /* empty */
#define _Py_ANNOTATE_PCQ_GET(pcq) /* empty */
#define _Py_ANNOTATE_NEW_MEMORY(address, size) /* empty */
#define _Py_ANNOTATE_EXPECT_RACE(address, description) /* empty */
#define _Py_ANNOTATE_BENIGN_RACE(address, description) /* empty */
#define _Py_ANNOTATE_BENIGN_RACE_SIZED(address, size, description) /* empty */
#define _Py_ANNOTATE_PURE_HAPPENS_BEFORE_MUTEX(mu) /* empty */
#define _Py_ANNOTATE_MUTEX_IS_USED_AS_CONDVAR(mu) /* empty */
#define _Py_ANNOTATE_TRACE_MEMORY(arg) /* empty */
#define _Py_ANNOTATE_THREAD_NAME(name) /* empty */
#define _Py_ANNOTATE_IGNORE_READS_BEGIN() /* empty */
#define _Py_ANNOTATE_IGNORE_READS_END() /* empty */
#define _Py_ANNOTATE_IGNORE_WRITES_BEGIN() /* empty */
#define _Py_ANNOTATE_IGNORE_WRITES_END() /* empty */
#define _Py_ANNOTATE_IGNORE_READS_AND_WRITES_BEGIN() /* empty */
#define _Py_ANNOTATE_IGNORE_READS_AND_WRITES_END() /* empty */
#define _Py_ANNOTATE_IGNORE_SYNC_BEGIN() /* empty */
#define _Py_ANNOTATE_IGNORE_SYNC_END() /* empty */
#define _Py_ANNOTATE_ENABLE_RACE_DETECTION(enable) /* empty */
#define _Py_ANNOTATE_NO_OP(arg) /* empty */
#define _Py_ANNOTATE_FLUSH_STATE() /* empty */

#endif  /* DYNAMIC_ANNOTATIONS_ENABLED */

/* Use the macros above rather than using these functions directly. */
#ifdef __cplusplus
extern "C" {
#endif
void AnnotateRWLockCreate(const char *file, int line,
                          const volatile void *lock);
void AnnotateRWLockDestroy(const char *file, int line,
                           const volatile void *lock);
void AnnotateRWLockAcquired(const char *file, int line,
                            const volatile void *lock, long is_w);
void AnnotateRWLockReleased(const char *file, int line,
                            const volatile void *lock, long is_w);
void AnnotateBarrierInit(const char *file, int line,
                         const volatile void *barrier, long count,
                         long reinitialization_allowed);
void AnnotateBarrierWaitBefore(const char *file, int line,
                               const volatile void *barrier);
void AnnotateBarrierWaitAfter(const char *file, int line,
                              const volatile void *barrier);
void AnnotateBarrierDestroy(const char *file, int line,
                            const volatile void *barrier);
void AnnotateCondVarWait(const char *file, int line,
                         const volatile void *cv,
                         const volatile void *lock);
void AnnotateCondVarSignal(const char *file, int line,
                           const volatile void *cv);
void AnnotateCondVarSignalAll(const char *file, int line,
                              const volatile void *cv);
void AnnotatePublishMemoryRange(const char *file, int line,
                                const volatile void *address,
                                long size);
void AnnotateUnpublishMemoryRange(const char *file, int line,
                                  const volatile void *address,
                                  long size);
void AnnotatePCQCreate(const char *file, int line,
                       const volatile void *pcq);
void AnnotatePCQDestroy(const char *file, int line,
                        const volatile void *pcq);
void AnnotatePCQPut(const char *file, int line,
                    const volatile void *pcq);
void AnnotatePCQGet(const char *file, int line,
                    const volatile void *pcq);
void AnnotateNewMemory(const char *file, int line,
                       const volatile void *address,
                       long size);
void AnnotateExpectRace(const char *file, int line,
                        const volatile void *address,
                        const char *description);
void AnnotateBenignRace(const char *file, int line,
                        const volatile void *address,
                        const char *description);
void AnnotateBenignRaceSized(const char *file, int line,
                        const volatile void *address,
                        long size,
                        const char *description);
void AnnotateMutexIsUsedAsCondVar(const char *file, int line,
                                  const volatile void *mu);
void AnnotateTraceMemory(const char *file, int line,
                         const volatile void *arg);
void AnnotateThreadName(const char *file, int line,
                        const char *name);
void AnnotateIgnoreReadsBegin(const char *file, int line);
void AnnotateIgnoreReadsEnd(const char *file, int line);
void AnnotateIgnoreWritesBegin(const char *file, int line);
void AnnotateIgnoreWritesEnd(const char *file, int line);
void AnnotateEnableRaceDetection(const char *file, int line, int enable);
void AnnotateNoOp(const char *file, int line,
                  const volatile void *arg);
void AnnotateFlushState(const char *file, int line);

/* Return non-zero value if running under valgrind.

  If "valgrind.h" is included into dynamic_annotations.c,
  the regular valgrind mechanism will be used.
  See http://valgrind.org/docs/manual/manual-core-adv.html about
  RUNNING_ON_VALGRIND and other valgrind "client requests".
  The file "valgrind.h" may be obtained by doing
     svn co svn://svn.valgrind.org/valgrind/trunk/include

  If for some reason you can't use "valgrind.h" or want to fake valgrind,
  there are two ways to make this function return non-zero:
    - Use environment variable: export RUNNING_ON_VALGRIND=1
    - Make your tool intercept the function RunningOnValgrind() and
      change its return value.
 */
int RunningOnValgrind(void);

#ifdef __cplusplus
}
#endif

#if DYNAMIC_ANNOTATIONS_ENABLED != 0 && defined(__cplusplus)

  /* _Py_ANNOTATE_UNPROTECTED_READ is the preferred way to annotate racey reads.

     Instead of doing
        _Py_ANNOTATE_IGNORE_READS_BEGIN();
        ... = x;
        _Py_ANNOTATE_IGNORE_READS_END();
     one can use
        ... = _Py_ANNOTATE_UNPROTECTED_READ(x); */
  template <class T>
  inline T _Py_ANNOTATE_UNPROTECTED_READ(const volatile T &x) {
    _Py_ANNOTATE_IGNORE_READS_BEGIN();
    T res = x;
    _Py_ANNOTATE_IGNORE_READS_END();
    return res;
  }
  /* Apply _Py_ANNOTATE_BENIGN_RACE_SIZED to a static variable. */
#define _Py_ANNOTATE_BENIGN_RACE_STATIC(static_var, description)        \
    namespace {                                                       \
      class static_var ## _annotator {                                \
       public:                                                        \
        static_var ## _annotator() {                                  \
          _Py_ANNOTATE_BENIGN_RACE_SIZED(&static_var,                     \
                                      sizeof(static_var),             \
            # static_var ": " description);                           \
        }                                                             \
      };                                                              \
      static static_var ## _annotator the ## static_var ## _annotator;\
    }
#else /* DYNAMIC_ANNOTATIONS_ENABLED == 0 */

#define _Py_ANNOTATE_UNPROTECTED_READ(x) (x)
#define _Py_ANNOTATE_BENIGN_RACE_STATIC(static_var, description)  /* empty */

#endif /* DYNAMIC_ANNOTATIONS_ENABLED */

#endif  /* __DYNAMIC_ANNOTATIONS_H__ */
