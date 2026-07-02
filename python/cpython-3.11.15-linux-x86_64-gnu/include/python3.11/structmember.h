#ifndef Py_STRUCTMEMBER_H
#define Py_STRUCTMEMBER_H
#ifdef __cplusplus
extern "C" {
#endif


/* Interface to map C struct members to Python object attributes */

#include <stddef.h> /* For offsetof */

/* An array of PyMemberDef structures defines the name, type and offset
   of selected members of a C structure.  These can be read by
   PyMember_GetOne() and set by PyMember_SetOne() (except if their READONLY
   flag is set).  The array must be terminated with an entry whose name
   pointer is NULL. */

struct PyMemberDef {
    const char *name;
    int type;
    Py_ssize_t offset;
    int flags;
    const char *doc;
};

/* Types */
#define T_SHORT     0
#define T_INT       1
#define T_LONG      2
#define T_FLOAT     3
#define T_DOUBLE    4
#define T_STRING    5
#define T_OBJECT    6
/* XXX the ordering here is weird for binary compatibility */
#define T_CHAR      7   /* 1-character string */
#define T_BYTE      8   /* 8-bit signed int */
/* unsigned variants: */
#define T_UBYTE     9
#define T_USHORT    10
#define T_UINT      11
#define T_ULONG     12

/* Added by Jack: strings contained in the structure */
#define T_STRING_INPLACE    13

/* Added by Lillo: bools contained in the structure (assumed char) */
#define T_BOOL      14

#define T_OBJECT_EX 16  /* Like T_OBJECT, but raises AttributeError
                           when the value is NULL, instead of
                           converting to None. */
#define T_LONGLONG      17
#define T_ULONGLONG     18

#define T_PYSSIZET      19      /* Py_ssize_t */
#define T_NONE          20      /* Value is always None */


/* Flags */
#define READONLY            1
#define READ_RESTRICTED     2
#define PY_WRITE_RESTRICTED 4
#define RESTRICTED          (READ_RESTRICTED | PY_WRITE_RESTRICTED)

#define PY_AUDIT_READ       READ_RESTRICTED

/* Current API, use this */
PyAPI_FUNC(PyObject *) PyMember_GetOne(const char *, PyMemberDef *);
PyAPI_FUNC(int) PyMember_SetOne(char *, PyMemberDef *, PyObject *);


#ifdef __cplusplus
}
#endif
#endif /* !Py_STRUCTMEMBER_H */
